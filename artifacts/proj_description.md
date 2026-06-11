# FEATURE — Comparing wav2vec and wav2vec 2.0 for Phonological Feature Representation Across Languages

**Main contact:** Nina Goes, [nina.goes@fau.de](mailto:nina.goes@fau.de)

## TL;DR

This project compares different wav2vec and wav2vec 2.0 based speech representation models to study how well they capture phonological features across languages. It investigates whether multilingual and cross-lingual pretrained models encode properties such as voicing, nasality, place of articulation, and manner of articulation in a language-independent way, and how these representations can be improved or adapted through fine-tuning.

## Background

Self-supervised speech models such as wav2vec 2.0 have become important foundations for speech processing because they learn strong representations directly from raw waveform audio. Multilingual variants such as XLSR and XLS-R extend this idea across many languages and have shown strong cross-lingual transfer performance. Prior probing work also suggests that wav2vec 2.0 embeddings encode substantial phonetic and phonological information, making them a good basis for studying phonological feature representations across languages.

## Overall Goal

The goal of this project is to systematically compare different wav2vec / wav2vec 2.0 model variants and evaluate how well phonological features are represented across different languages, and whether certain languages share latent phonological features.

## Tasks

- Establish multilingual speech data pipeline
- Select wav2vec / wav2vec 2.0 model variants
- Extract hidden representations from different model layers
- Train probing models for phonological features
- Compare representation quality across languages
- Compare monolingual and multilingual pretrained models
- Analyze which layers encode which features best
- Evaluate cross-lingual transferability of phonological representations

> This project evaluates how wav2vec models represent phonological features (like voicing and articulation) across different languages. By comparing multilingual and cross-lingual pretraining, the study explores whether these speech properties are encoded universally and how fine-tuning can better adapt these representations.

## Models and Dataset

**Pretrained model comparison:** wav2vec 2.0 base vs XLSR / multilingual wav2vec 2.0

- [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
- [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)
- [facebook/wav2vec2-xls-r-300](https://huggingface.co/facebook/wav2vec2-xls-r-300)

**Dataset:**

- [FLEURS](https://huggingface.co/datasets/google/fleurs): Goyal, Naman, et al. "The Flores-101 evaluation benchmark for low-resource and multilingual machine translation." *Transactions of the Association for Computational Linguistics* 10 (2022): 522–538.

## What Can We Analyze?

- Which pretrained model best captures language-independent phonological features?
- Are multilingual models better than monolingual ones for cross-lingual phonological probing?
- Which layers encode phonological information most strongly?
- Which phonological features transfer well across languages, and which are language-specific?

## Goal and MVP



**Overall goal:** The goal of this project is to systematically compare different wav2vec / wav2vec 2.0 model variants and evaluate how well phonological features are represented across different languages and if certain languages share latent phonological features.
