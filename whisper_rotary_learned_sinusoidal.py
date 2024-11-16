## Whisper with rotary encoder and learned-sinusoid rotary decoder

# import multiprocessing
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as at
from torch.utils.data import Dataset
#from data_loader import eval_dataloader, train_dataloader
import torch.utils.data as Data
import base64, csv, torchaudio, neologdn, evaluate, MeCab, gzip, numpy as np, torch.nn.functional as F, torch, whisper
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
from transformers import TrainingArguments, WhisperTokenizer
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from rotary_embedding_torch import RotaryEmbedding
from transformers import Trainer, WhisperFeatureExtractor, WhisperTokenizerFast
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
from whisper import load_audio, log_mel_spectrogram, pad_or_trim
from typing import Any, Dict, List, Union
from torchaudio import datasets
from tqdm import tqdm
from torch.optim import AdamW, lr_scheduler
from torch import optim
import torch.utils.checkpoint as checkpoint
from decoding import decode as decode_function
from decoding import detect_language as detect_language_function
from transcribe import transcribe as transcribe_function
from transformers import Adafactor
from tqdm import tqdm
import torch, os, time
import torch.optim as optim
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.profiler as profiler

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

import torch
from transformers import WhisperForConditionalGeneration



@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

class LayerNorm(nn.Module): #RMSNorm
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma
    
class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )
    
def sinusofeatures(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state

class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_state // n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.rotary_emb = RotaryEmbedding(dim=n_state // n_head)

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        q = self.query(x)
        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and n_ctx > 1)
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()
        return out, qk

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attention = cross_attention
        if self.cross_attention:
            self.cross_attn = MultiHeadAttention(n_state, n_head)
            self.cross_attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attention:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: torch.Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = checkpoint.checkpoint(block, x)
        
        x = self.ln_post(x)
        return x

def block_forward(block, x, xa, mask, kv_cache):
    return block(x, xa, mask=mask, kv_cache=kv_cache)

class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = LearnedSinusoidalEmbeddings(n_ctx, n_state)
        self.rotary_emb = RotaryEmbedding(dim=n_state // n_head)
        
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer('mask', mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        positions = torch.arange(offset, offset + x.shape[-1], device=x.device)
        x = self.token_embedding(x) + self.positional_embedding(positions)
        x = x.to(xa.dtype)

        # for block in self.blocks:
        #     x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        for block in self.blocks:
            x = checkpoint.checkpoint(block_forward, block, x, xa, self.mask, kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()
        return logits

   
class LearnedSinusoidalEmbeddings(nn.Module): # sinusofeatures(n_ctx, n_state)
    def __init__(self, n_ctx, n_state):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_state = n_state

        sinusoidal_embeddings = sinusofeatures(n_ctx, n_state)
        self.positional_embeddings = nn.Parameter(sinusoidal_embeddings)

    def forward(self, positions):
        position_embeddings = self.positional_embeddings[positions]
        return position_embeddings

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer('alignment_heads', all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer('alignment_heads', mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)
        logits = self.decoder(tokens, audio_features)
        return {"logits": logits}

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

dimensions = ModelDimensions(
    n_mels=80, 
    n_audio_ctx=1500, 
    n_audio_state=1024, 
    n_audio_head=16, 
    n_audio_layer=8, 
    n_vocab=51865, 
    n_text_ctx=448, 
    n_text_state=1024, 
    n_text_head=16, 
    n_text_layer=8
    )

pretrained_model = whisper.load_model("medium")
pretrained_state_dict = pretrained_model.state_dict()

model = Whisper(dimensions).cuda()
model_state_dict = model.state_dict()

def transfer_layer(src_name, tgt_name):
    if src_name in pretrained_state_dict and tgt_name in model_state_dict:
        src_tensor = pretrained_state_dict[src_name]
        tgt_tensor = model_state_dict[tgt_name]
        print(f'Transferring layer {src_name} to {tgt_name}')
        print(f'Source shape: {src_tensor.shape}, Target shape: {tgt_tensor.shape}')
        tgt_tensor.copy_(src_tensor)

# Transfer convolutional layers
transfer_layer('model.encoder.conv1.weight', 'encoder.conv1.weight')
transfer_layer('model.encoder.conv1.bias', 'encoder.conv1.bias')
transfer_layer('model.encoder.conv2.weight', 'encoder.conv2.weight')
transfer_layer('model.encoder.conv2.bias', 'encoder.conv2.bias')

# Transfer layer norms (skip custom LayerNorm)
# transfer_layer('model.encoder.layer_norm.weight', 'encoder.ln_post.weight')
# transfer_layer('model.encoder.layer_norm.bias', 'encoder.ln_post.bias')
# transfer_layer('model.decoder.layer_norm.weight', 'decoder.ln.weight')
# transfer_layer('model.decoder.layer_norm.bias', 'decoder.ln.bias')

# Transfer multi-head attention layers (skip custom layers)
for i in range(12):  # Adjust according to actual layer count
    transfer_layer(f'model.encoder.blocks.{i}.attn.query.weight', f'encoder.blocks.{i}.query.weight')
    transfer_layer(f'model.encoder.blocks.{i}.attn.query.bias', f'encoder.blocks.{i}.query.bias')
    transfer_layer(f'model.encoder.blocks.{i}.attn.key.weight', f'encoder.blocks.{i}.key.weight')
    transfer_layer(f'model.encoder.blocks.{i}.attn.value.weight', f'encoder.blocks.{i}.value.weight')
    transfer_layer(f'model.encoder.blocks.{i}.attn.out.weight', f'encoder.blocks.{i}.out.weight')
    transfer_layer(f'model.encoder.blocks.{i}.attn.out.bias', f'encoder.blocks.{i}.out.bias')
    transfer_layer(f'model.decoder.blocks.{i}.attn.query.weight', f'decoder.blocks.{i}.query.weight')
    transfer_layer(f'model.decoder.blocks.{i}.attn.query.bias', f'decoder.blocks.{i}.query.bias')
    transfer_layer(f'model.decoder.blocks.{i}.attn.key.weight', f'decoder.blocks.{i}.key.weight')
    transfer_layer(f'model.decoder.blocks.{i}.attn.value.weight', f'decoder.blocks.{i}.value.weight')
    transfer_layer(f'model.decoder.blocks.{i}.attn.out.weight', f'decoder.blocks.{i}.out.weight')
    transfer_layer(f'model.decoder.blocks.{i}.attn.out.bias', f'decoder.blocks.{i}.out.bias')
    
# Load the modified state dict into the new model
model.load_state_dict(model_state_dict)
tokenizer = WhisperTokenizerFast.from_pretrained("D:/proj/models/new_whisper_medium", task="transcribe", language="japanese", local_files_only=True)
csv_file = 'D:/proj/datasets/gv_test/metadata.csv'
audio_dir = 'D:/proj/datasets/gv_test/'


def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform

class CustomAudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, tokenizer, sample_rate=16000):
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.samples = []

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row if it exists
            for row in reader:
                audio_path, label = row[0], row[1]
                self.samples.append((audio_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        audio_path = f'{self.audio_dir}/{audio_path}'

        audio = load_wave(audio_path, sample_rate=self.sample_rate)

        return {
            'audio': audio,
            'label': label
        }

dataset = CustomAudioDataset(csv_file, audio_dir, tokenizer)

def collate_fn(batch):
    input_features, labels, dec_input_features = [], [], []
    
    for f in batch:
        # Convert audio to features here
        audio = whisper.pad_or_trim(f["audio"].flatten())
        input_feature = whisper.log_mel_spectrogram(audio, n_mels=80)

        label = f["label"]
        label_tokens = [tokenizer.bos_token_id] + tokenizer.encode(label) + [tokenizer.eos_token_id]
        dec_input_feature = label_tokens[:-1]
        label = label_tokens[1:]

        input_features.append(input_feature)
        labels.append(label)
        dec_input_features.append(dec_input_feature)

    input_features = torch.stack(input_features)

    max_label_len = max(len(l) for l in labels)
    max_dec_input_len = max(len(d) for d in dec_input_features)
    max_len = max(max_label_len, max_dec_input_len)

    labels = [np.pad(l, (0, max_len - len(l)), 'constant', constant_values=-100) for l in labels]
    dec_input_features = [np.pad(d, (0, max_len - len(d)), 'constant', constant_values=tokenizer.pad_token_id) for d in dec_input_features]

    # Convert the lists of numpy arrays to numpy arrays before creating tensors
    labels = np.array(labels)
    dec_input_features = np.array(dec_input_features)

    labels = torch.tensor(labels, dtype=torch.long)
    dec_input_features = torch.tensor(dec_input_features, dtype=torch.long)

    batch = {
        "input_features": input_features,
        "labels": labels,
        "dec_input_features": dec_input_features
    }
    return batch

metrics_cer = evaluate.load("cer")
metrics_wer = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred["predictions"]
    label_ids = pred["label_ids"]
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metrics_cer.compute(predictions=pred_str, references=label_str)
    wer = 100 * metrics_wer.compute(predictions=pred_str, references=label_str)
    return {"cer": cer, "wer": wer}



# def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
#     waveform, sr = torchaudio.load(wave_path, normalize=True)
#     if sample_rate != sr:
#         waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
#     return waveform

# class CustomAudioDataset(Dataset):
#     def __init__(self, csv_file, audio_dir, tokenizer, sample_rate=16000):
#         self.audio_dir = audio_dir
#         self.tokenizer = tokenizer
#         self.sample_rate = sample_rate
#         self.samples = []

#         with open(csv_file, 'r', encoding='utf-8') as f:
#             reader = csv.reader(f)
#             next(reader)  # Skip header row if it exists
#             for row in reader:
#                 audio_path, label = row[0], row[1]
#                 self.samples.append((audio_path, label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         audio_path, label = self.samples[idx]
#         audio_path = f'{self.audio_dir}/{audio_path}'

#         audio = load_wave(audio_path, sample_rate=self.sample_rate)
#         audio = whisper.pad_or_trim(audio.flatten())
#         input_features = whisper.log_mel_spectrogram(audio, n_mels=80)

#         label_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(label) + [self.tokenizer.eos_token_id]
#         dec_input_features = label_tokens[:-1]
#         labels = label_tokens[1:]

#         return {
#             'input_features': input_features,
#             'dec_input_features': dec_input_features,
#             'labels': labels
#         }

# dataset = CustomAudioDataset(csv_file, audio_dir, tokenizer)

# def collate_fn(batch):
#     input_features, labels, dec_input_features = [], [], []
#     for f in batch:
#         input_features.append(f["input_features"])
#         labels.append(f["labels"])
#         dec_input_features.append(f["dec_input_features"])

#     input_features = torch.stack(input_features)

#     max_label_len = max(len(l) for l in labels)
#     max_dec_input_len = max(len(d) for d in dec_input_features)
#     max_len = max(max_label_len, max_dec_input_len)

#     labels = [np.pad(l, (0, max_len - len(l)), 'constant', constant_values=-100) for l in labels]
#     dec_input_features = [np.pad(d, (0, max_len - len(d)), 'constant', constant_values=tokenizer.pad_token_id) for d in dec_input_features]

#     # Convert the lists of numpy arrays to numpy arrays before creating tensors
#     labels = np.array(labels)
#     dec_input_features = np.array(dec_input_features)

#     labels = torch.tensor(labels, dtype=torch.long)
#     dec_input_features = torch.tensor(dec_input_features, dtype=torch.long)

#     batch = {
#         "input_features": input_features,
#         "labels": labels,
#         "dec_input_features": dec_input_features
#     }
#     return batch

# metrics_cer = evaluate.load("cer")
# metrics_wer = evaluate.load("wer")

# def compute_metrics(pred):
#     pred_ids = pred["predictions"]
#     label_ids = pred["label_ids"]
#     label_ids[label_ids == -100] = tokenizer.pad_token_id
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

#     cer = 100 * metrics_cer.compute(predictions=pred_str, references=label_str)
#     wer = 100 * metrics_wer.compute(predictions=pred_str, references=label_str)
#     return {"cer": cer, "wer": wer}


def train_val_dataset(dataset, val_split=0.001):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

datasets = train_val_dataset(dataset, val_split=0.001)
train_dataset = datasets['train']
eval_dataset = datasets['val']

# Functions to create DataLoaders
def train_dataloader():   
    return DataLoader(
        train_dataset,
        batch_size=1,  # Adjust batch size as needed
        drop_last=False, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn
    )

def eval_dataloader():
    return DataLoader(
        eval_dataset,
        batch_size=1,  # Adjust batch size as needed
        drop_last=False,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
import logging
import os
from torch.utils.tensorboard import SummaryWriter

checkpoint_dir = 'D:/proj/models/ckpt/'
os.makedirs(checkpoint_dir, exist_ok=True)
log_dir = 'D:/proj/models/ckpt/logs/'
os.makedirs(log_dir, exist_ok=True)

# Create a SummaryWriter for TensorBoard, saving logs to the specified directory
writer = SummaryWriter(log_dir)

# Set up logging to a file
logging.basicConfig(
    filename=os.path.join(log_dir, 'training.log'), 
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

# Verify the dataset
print(f"Number of samples in dataset: {len(dataset)}")
for i in range(min(3, len(dataset))):  # Print first few samples for verification
    print(f"Sample {i}: {dataset[i]}")
# Create directories for checkpoints and logs
checkpoint_dir = 'D:/proj/models/ckpt/'
os.makedirs(checkpoint_dir, exist_ok=True)
log_dir = 'D:/proj/models/ckpt/logs/'
os.makedirs(log_dir, exist_ok=True)

# Create a SummaryWriter for TensorBoard
writer = SummaryWriter(log_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'training.log'), 
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

def train_and_evaluate(model, train_loader, eval_loader, optimizer, loss_fn, scheduler, num_epochs=1, device='cuda', accumulation_steps=1, clear_cache=True, log_interval=5, eval_interval=100, save_interval=100, checkpoint_dir=checkpoint_dir, log_dir=log_dir):
    model.to(device)
    global_step = 0
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].long().to(device)
            dec_input_features = batch['dec_input_features'].to(device)

            with autocast(device_type="cuda", dtype=torch.float16):
                # Forward pass
                encoder_outputs = model.encoder(input_features)
                decoder_outputs = model.decoder(dec_input_features, encoder_outputs)
                logits = decoder_outputs.view(-1, decoder_outputs.size(-1))
                loss = loss_fn(logits, labels.view(-1))
                total_loss += loss.item()
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            # Perform optimization step every accumulation_steps
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Optionally clear cache
                if clear_cache:
                    torch.cuda.empty_cache()

            global_step += 1

            # Logging at specified intervals
            if global_step % log_interval == 0:
                writer.add_scalar('Loss/train', total_loss / (step + 1), global_step)
                logging.info(f"Step {global_step}, Loss: {total_loss / (step + 1)}")

            # Evaluate at specified intervals
            if global_step % eval_interval == 0:
                model.eval()
                eval_loss = 0
                all_predictions = []
                all_labels = []
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        input_features = eval_batch['input_features'].to(device)
                        labels = eval_batch['labels'].long().to(device)
                        dec_input_features = eval_batch['dec_input_features'].to(device)

                        encoder_outputs = model.encoder(input_features)
                        decoder_outputs = model.decoder(dec_input_features, encoder_outputs)

                        # Compute loss
                        logits = decoder_outputs.view(-1, decoder_outputs.size(-1))
                        loss = loss_fn(logits, labels.view(-1))
                        eval_loss += loss.item()

                        all_predictions.extend(torch.argmax(decoder_outputs, dim=-1).cpu().numpy().tolist())
                        all_labels.extend(labels.cpu().numpy().tolist())

                # Prepare for metric computation
                predictions = {
                    "predictions": np.array(all_predictions),
                    "label_ids": np.array(all_labels)
                }

                # Compute metrics
                metrics = compute_metrics(predictions)
                writer.add_scalar('Loss/eval', eval_loss / len(eval_loader), global_step)
                writer.add_scalar('CER', metrics['cer'], global_step)

                print(f"Step {global_step}, Eval Loss: {eval_loss / len(eval_loader)}, CER: {metrics['cer']}, WER: {metrics['wer']}")
                logging.info(f"Step {global_step}, Eval Loss: {eval_loss / len(eval_loader)}, CER: {metrics['cer']}, WER: {metrics['wer']}")

                # Step the scheduler based on evaluation loss
                scheduler.step(eval_loss / len(eval_loader))

                # Print sample predictions and labels for verification
                sample_indices = range(min(3, len(all_predictions)))  # Print up to 3 sample predictions
                for idx in sample_indices:
                    pred_str = tokenizer.decode(all_predictions[idx], skip_special_tokens=True)
                    label_str = tokenizer.decode(all_labels[idx], skip_special_tokens=True)
                    print(f"Sample {idx}: Prediction: {pred_str}, Label: {label_str}")
                    logging.info(f"Sample {idx}: Prediction: {pred_str}, Label: {label_str}")

                model.train()

            # Save model at specified intervals
            if global_step % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved at step {global_step} to {checkpoint_path}")
                logging.info(f"Model saved at step {global_step} to {checkpoint_path}")

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
        logging.info(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    logging.info(f"Final model saved to {final_model_path}")

# Optimizer setup
optimizer = optim.Adafactor(
    model.parameters(), 
    lr=0.025, 
    beta2_decay=-0.8, 
    eps=(None, 0.001), 
    d=1.0, 
    weight_decay=0.0, 
    foreach=None, 
    maximize=False
)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

# Scheduler setup
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10,
    threshold=0.0001,
    threshold_mode='rel',
    cooldown=0,
    min_lr=0,
    eps=1e-08
)

# Create DataLoaders
train_loader = train_dataloader()
eval_loader = eval_dataloader()

# Train and evaluate the model with logging, evaluation, and model saving
train_and_evaluate(model, train_loader, eval_loader, optimizer, loss_fn, scheduler, num_epochs=1, device='cuda', accumulation_steps=1, clear_cache=True, log_interval=10, eval_interval=20, save_interval=100, checkpoint_dir=checkpoint_dir, log_dir=log_dir)
