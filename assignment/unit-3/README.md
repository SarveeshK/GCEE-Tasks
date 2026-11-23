# Unit 3: English to Hindi Neural Machine Translation using Transformer

## Overview
This module implements an Encoder Decoder Transformer architecture from scratch in PyTorch for English to Hindi machine translation without using pretrained model weights or pretrained tokenizers. All training logs, metric curves, BLEU comparisons, and qualitative translations are embedded inside `Transformer_Translation_Model.ipynb`.

## Dataset and Tokenization
* **Dataset:** `Aarif1430/english-to-hindi` (~128,000 parallel English and Hindi sentence pairs)
* **Filtering and Split:** Sentences filtered up to 40 words and partitioned into 90% Training, 5% Validation, and 5% Test sets
* **Custom Subword Tokenizers:** Separate 12,000 token Byte Pair Encoding (BPE) vocabularies trained from scratch on the training split using `[PAD]`, `[UNK]`, `[SOS]`, and `[EOS]` control tokens

## Model and Training Setup
* **Architecture:** Encoder Decoder Transformer (`d_model=256`, `dim_feedforward=1024`, `dropout=0.1`) with sinusoidal positional encoding, causal decoder masking, and Xavier Uniform weight initialization
* **Optimization:** Trained for 12 epochs using Teacher Forcing, Cross Entropy Loss with Label Smoothing (`0.1`), Adam optimizer (`lr=5e-4`, `betas=(0.9, 0.98)`), Linear Warmup with Cosine Decay, and Gradient Clipping (`max_norm=1.0`)

## Experimental Results

| Configuration | Parameters | Model Size (MB) | Best Val Loss | Best Val Acc (%) | Best Val PPL | Test BLEU (300 Samples) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A (2L, 4H) — Greedy** | 12,915,424 | 49.8 | 3.9671 | 47.16% | 52.83 | 10.67 |
| **Variant B (4L, 8H) — Greedy** | 16,601,824 | 63.9 | 3.8827 | 48.69% | 48.56 | 11.54 |
| **Variant B (4L, 8H) — Beam Search (k=3)** | 16,601,824 | 63.9 | 3.8827 | 48.69% | 48.56 | **13.40** |

## Observations
* **Architectural Depth:** Increasing depth from 2 to 4 layers and attention heads from 4 to 8 consistently lowered validation perplexity (52.83 to 48.56) and improved next-token accuracy (47.16% to 48.69%).
* **Decoding Strategy:** Length-normalized Beam Search (`beam_width=3`, `alpha=0.6`) outperformed Greedy decoding by +1.86 BLEU points on unseen test sentences by exploring multiple candidate hypotheses instead of committing greedily at each step.

## Files
* `Transformer_Translation_Model.ipynb`: Complete PyTorch implementation, training curves, BLEU benchmark chart, and sample English to Hindi translations
