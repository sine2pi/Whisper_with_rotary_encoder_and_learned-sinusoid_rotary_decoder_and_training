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

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        # for block in self.blocks:
        #     x = checkpoint.checkpoint(block_forward, block, x, xa, self.mask, kv_cache)

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
    n_mels=128, 
    n_audio_ctx=1500, 
    n_audio_state=1280, 
    n_audio_head=20, 
    n_audio_layer=24, 
    n_vocab=51866, 
    n_text_ctx=448, 
    n_text_state=1280, 
    n_text_head=16, 
    n_text_layer=4
    )

model = Whisper(dimensions).cuda()


pretrained_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3-turbo')
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

transfer_layer('model.encoder.conv1.weight', 'encoder.conv1.weight')
transfer_layer('model.encoder.conv1.bias', 'encoder.conv1.bias')
transfer_layer('model.encoder.conv2.weight', 'encoder.conv2.weight')
transfer_layer('model.encoder.conv2.bias', 'encoder.conv2.bias')

# transfer_layer('model.encoder.embed_positions.weight', 'encoder.positional_embedding.weight')
# transfer_layer('model.decoder.embed_positions.weight', 'decoder.positional_embedding.weight')

transfer_layer('model.encoder.layer_norm.weight', 'encoder.ln_post.weight')
transfer_layer('model.encoder.layer_norm.bias', 'encoder.ln_post.bias')
transfer_layer('model.decoder.layer_norm.weight', 'decoder.ln.weight')
transfer_layer('model.decoder.layer_norm.bias', 'decoder.ln.bias')

transfer_layer('model.decoder.embed_tokens.weight', 'decoder.token_embedding.weight') 

for i in range(6):
    transfer_layer(f'model.encoder.layers.{i}.self_attn.k_proj.weight', f'encoder.blocks.{i}.attn.key.weight')
    transfer_layer(f'model.encoder.layers.{i}.self_attn.v_proj.weight', f'encoder.blocks.{i}.attn.value.weight')
    transfer_layer(f'model.encoder.layers.{i}.self_attn.q_proj.weight', f'encoder.blocks.{i}.attn.query.weight')
    transfer_layer(f'model.encoder.layers.{i}.self_attn.out_proj.weight', f'encoder.blocks.{i}.attn.out.weight')
    transfer_layer(f'model.encoder.layers.{i}.self_attn_layer_norm.weight', f'encoder.blocks.{i}.attn_ln.weight')
    transfer_layer(f'model.encoder.layers.{i}.self_attn_layer_norm.bias', f'encoder.blocks.{i}.attn_ln.bias')
    transfer_layer(f'model.encoder.layers.{i}.fc1.weight', f'encoder.blocks.{i}.mlp.0.weight')
    transfer_layer(f'model.encoder.layers.{i}.fc1.bias', f'encoder.blocks.{i}.mlp.0.bias')
    transfer_layer(f'model.encoder.layers.{i}.fc2.weight', f'encoder.blocks.{i}.mlp.2.weight')
    transfer_layer(f'model.encoder.layers.{i}.fc2.bias', f'encoder.blocks.{i}.mlp.2.bias')
    transfer_layer(f'model.encoder.layers.{i}.final_layer_norm.weight', f'encoder.blocks.{i}.mlp_ln.weight')
    transfer_layer(f'model.encoder.layers.{i}.final_layer_norm.bias', f'encoder.blocks.{i}.mlp_ln.bias')
    transfer_layer(f'model.decoder.layers.{i}.self_attn.k_proj.weight', f'decoder.blocks.{i}.attn.key.weight')
    transfer_layer(f'model.decoder.layers.{i}.self_attn.v_proj.weight', f'decoder.blocks.{i}.attn.value.weight')
    transfer_layer(f'model.decoder.layers.{i}.self_attn.q_proj.weight', f'decoder.blocks.{i}.attn.query.weight')
    transfer_layer(f'model.decoder.layers.{i}.self_attn.out_proj.weight', f'decoder.blocks.{i}.attn.out.weight')
    transfer_layer(f'model.decoder.layers.{i}.self_attn_layer_norm.weight', f'decoder.blocks.{i}.attn_ln.weight')
    transfer_layer(f'model.decoder.layers.{i}.self_attn_layer_norm.bias', f'decoder.blocks.{i}.attn_ln.bias')
    transfer_layer(f'model.decoder.layers.{i}.encoder_attn.k_proj.weight', f'decoder.blocks.{i}.cross_attn.key.weight')
    transfer_layer(f'model.decoder.layers.{i}.encoder_attn.v_proj.weight', f'decoder.blocks.{i}.cross_attn.value.weight')
    transfer_layer(f'model.decoder.layers.{i}.encoder_attn.q_proj.weight', f'decoder.blocks.{i}.cross_attn.query.weight')
    transfer_layer(f'model.decoder.layers.{i}.encoder_attn.out_proj.weight', f'decoder.blocks.{i}.cross_attn.out.weight')
    transfer_layer(f'model.decoder.layers.{i}.encoder_attn_layer_norm.weight', f'decoder.blocks.{i}.cross_attn_ln.weight')
    transfer_layer(f'model.decoder.layers.{i}.encoder_attn_layer_norm.bias', f'decoder.blocks.{i}.cross_attn_ln.bias')
    transfer_layer(f'model.decoder.layers.{i}.fc1.weight', f'decoder.blocks.{i}.mlp.0.weight')
    transfer_layer(f'model.decoder.layers.{i}.fc1.bias', f'decoder.blocks.{i}.mlp.0.bias')
    transfer_layer(f'model.decoder.layers.{i}.fc2.weight', f'decoder.blocks.{i}.mlp.2.weight')
    transfer_layer(f'model.decoder.layers.{i}.fc2.bias', f'decoder.blocks.{i}.mlp.2.bias')
    transfer_layer(f'model.decoder.layers.{i}.final_layer_norm.weight', f'decoder.blocks.{i}.mlp_ln.weight')
    transfer_layer(f'model.decoder.layers.{i}.final_layer_norm.bias', f'decoder.blocks.{i}.mlp_ln.bias')

model.load_state_dict(model_state_dict)



tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-medium')
csv_file = 'D:/proj/datasets/gvj/trimmed/metadata.csv'
audio_dir = 'D:/proj/datasets/gvj/trimmed/'


class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_features, labels, dec_input_features = [], [], []
        for f in features:
            input_features.append(f["input_features"])
            labels.append(f["labels"])
            dec_input_features.append(f["dec_input_features"])

        input_features = torch.concat([input_id[None, :] for input_id in input_features])

        label_lengths = [len(lab) for lab in labels]
        dec_input_features_length = [len(e) for e in dec_input_features]
        max_label_len = max(label_lengths+dec_input_features_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_features = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_features, dec_input_features_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_features": dec_input_features
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_features"] = input_features

        return batch
    
collate_fn=WhisperDataCollatorWhithPadding()


def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform

class CustomAudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, tokenizer, sample_rate=16000):
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.samples = []

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                audio_path, label = row[0], row[1]
                self.samples.append((audio_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        audio_path = f'{self.audio_dir}/{audio_path}'

        audio = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        input_features = whisper.log_mel_spectrogram(audio, n_mels=128)

        label_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(label) + [self.tokenizer.eos_token_id]
        dec_input_features = label_tokens[:-1]
        labels = label_tokens[1:]

        return {
            'input_features': input_features,
            'dec_input_features': dec_input_features,
            'labels': labels
        }

dataset = CustomAudioDataset(csv_file, audio_dir, tokenizer)

def train_val_dataset(dataset, val_split=0.001):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

datasets = train_val_dataset(dataset, val_split=0.001)

train_dataset = datasets['train']
eval_dataset = datasets['val']

def train_dataloader():   
    dataset = train_dataset
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        drop_last=False, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )

def eval_dataloader():
    dataset = eval_dataset
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

metric = evaluate.load("cer")
wakati = MeCab.Tagger("-Owakati")

def compute_metrics(pred):
    pred_features = pred.predictions
    label_features = pred.label_features
    label_features[label_features == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_features, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_features, skip_special_tokens=True)
    
    pred_str_nj = [wakati.parse(pred) for pred in pred_str] 
    label_str_nj = [wakati.parse(label) for label in label_str] 
    pred_str_nj = [pred_str_nj[i] for i in range(len(pred_str_nj)) if len(label_str_nj[i]) > 0]
    label_str_nj = [
        label_str_nj[i]
        for i in range(len(label_str_nj))
        if len(label_str_nj[i]) > 0]
    
    pred_str_neo = [neologdn.normalize(pred) for pred in pred_str] 
    label_str_neo = [neologdn.normalize(label) for label in label_str] 
    pred_str_neo = [pred_str_neo[i] for i in range(len(pred_str_neo)) if len(label_str_neo[i]) > 0]
    label_str_neo = [
        label_str_neo[i]
        for i in range(len(label_str_neo))
        if len(label_str_neo[i]) > 0]
    
    cer = 100 * metric.compute(predictions=pred_str, references=label_str) 
    cer_mecab = 100 * metric.compute(predictions=pred_str_nj, references=label_str_nj)
    cer_neo = 100 * metric.compute(predictions=pred_str_neo, references=label_str_neo) # 
    return {"cer": cer,  "cer_mecab": cer_mecab, "cer_neo": cer_neo}


def train_with_profiling(model, train_dataloader, eval_dataloader, criterion, num_epochs=1, device='cuda', accumulation_steps=2, eval_steps=10, clear_cache=True, checkpoint_dir="D:/proj/models/ckpt", checkpoint_interval=50, max_steps=100, steps_per_speed_update=10):
    model.to(device)
    model.train()

    optimizer = Adafactor(
    model.parameters(), 
    scale_parameter=True, 
    relative_step=True, 
    warmup_init=True, 
    lr=None
)
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()

        start_time = time.time()
        processed_samples_since_last_update = 0

        progress_bar = tqdm(train_dataloader(), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            if max_steps is not None and global_step >= max_steps:
                print("Reached max_steps. Stopping training.")
                return

            global_step += 1
            batch_size = batch['input_features'].size(0)
            processed_samples_since_last_update += batch_size

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_training"):
                    input_features = batch['input_features'].to(device)
                    labels = batch['labels'].long().to(device)
                    dec_input_features = batch['dec_input_features'].to(device)

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        encoder_outputs = model.encoder(input_features)
                        decoder_outputs = model.decoder(dec_input_features, encoder_outputs)

                        logits = decoder_outputs.view(-1, decoder_outputs.size(-1))
                        loss = criterion(logits, labels.view(-1))

                    total_loss += loss.item()
                    loss.backward()

                    if (step + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        if clear_cache:
                            torch.cuda.empty_cache()

                    if (step + 1) % eval_steps == 0:
                        model.eval()
                        with torch.no_grad():
                            all_predictions = []
                            all_labels = []
                            first_batch = True
                            for eval_batch in eval_dataloader():
                                if first_batch:
                                    print(f"First batch: Number of eval samples: {len(eval_batch['input_features'])}")
                                    first_batch = False

                                eval_input_features = eval_batch['input_features'].to(device)
                                eval_labels = eval_batch['labels'].long().to(device)
                                eval_dec_input_features = eval_batch['dec_input_features'].to(device)

                                encoder_outputs = model.encoder(eval_input_features)
                                decoder_outputs = model.decoder(eval_dec_input_features, encoder_outputs)

                                all_predictions.append(decoder_outputs)
                                all_labels.append(eval_labels)

                            all_predictions = pad_sequence(all_predictions, batch_first=True, padding_value=-100)
                            all_predictions = torch.cat([torch.argmax(p, dim=-1) for p in all_predictions], dim=0)
                            all_labels = torch.cat(all_labels, dim=0) 

                            metrics = compute_metrics({'predictions': all_predictions, 'label_features': all_labels})
                            print(f"Metrics at step {step + 1}, epoch {epoch + 1}: {metrics}")
                        model.train() 

                    if (step + 1) % checkpoint_interval == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_step_{step+1}.pt")
                        torch.save({
                            'epoch': epoch + 1,
                            'step': step + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                        }, checkpoint_path)
                        print(f"Checkpoint saved to {checkpoint_path}")

            if (step + 1) % steps_per_speed_update == 0:
                elapsed_time = time.time() - start_time
                samples_per_second = processed_samples_since_last_update / elapsed_time if elapsed_time > 0 else 0

                progress_bar.set_postfix(loss=total_loss / (step + 1), global_step=global_step, samples_per_second=f"{samples_per_second:.2f}")

                start_time = time.time()
                processed_samples_since_last_update = 0

        if (step + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(train_dataloader)}')

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

optimizer = optim.Adafactor(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

train_loader = train_dataloader()
eval_loader = eval_dataloader()

train_with_profiling(model, train_dataloader, eval_dataloader, criterion, num_epochs=1, device='cuda', accumulation_steps=2, eval_steps=10, clear_cache=True, checkpoint_dir="D:/proj/models/ckpt", checkpoint_interval=50, max_steps=100, steps_per_speed_update=10)



checkpoint_path = "D:/proj/models/ckpt/checkpoint_epoch_x_step_x.pt"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
start_step = checkpoint['step']
loss = checkpoint['loss']


