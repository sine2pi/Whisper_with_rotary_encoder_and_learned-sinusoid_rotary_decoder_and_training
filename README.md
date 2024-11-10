# Whisper_with_rotary_encoder_and_learned-sinusoid_rotary_decoder_and_training
Whisper with rotary encoder and learned-sinusoid-rotary decoder RMSnorm

My goal is to attempt to mitigate catastrophic forgetting in Whisper through different embedding schemes. By encoding positional information directly into the attention mechanism, rotary embeddings might allow the model to generalize better to different sequence lengths and potentially reduce the reliance on absolute positional information that could be specific to the pretraining data. Making the sinusoidal embeddings learnable allows them to adapt to the fine-tuning data, potentially reducing the conflict between the pretrained positional information and the new data.

Other strategies to consider:

Regularization Techniques:
L2 Regularization: Adding L2 regularization to the loss function can penalize large weight changes, encouraging the model to retain more of the pretrained knowledge.
Elastic Weight Consolidation (EWC): EWC identifies important parameters from the pretraining phase and adds a penalty term to the loss function that discourages changes to these parameters.
Synaptic Intelligence (SI): SI assigns importance weights to parameters based on their contribution to the pretraining task and uses these weights to regulate updates during fine-tuning.
