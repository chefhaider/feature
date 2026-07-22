#!/usr/bin/env python3
"""
Extract Wav2Vec2 embeddings from FLEURS dataset.
Usage: python extract_features.py --languages en_us de_de es_419 --max-samples 100
"""

import torch
import numpy as np
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import pickle
import os
import argparse
import logging
from typing import List, Dict, Any

# wav2vec2 models are trained on 16 kHz audio and do NOT resample internally.
TARGET_SR = 16000

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Wav2Vec2 hidden states from FLEURS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="facebook/wav2vec2-base",
        help="Wav2Vec2 model name from HuggingFace"
    )
    
    parser.add_argument(
        "--languages", "-l",
        nargs="+",
        default=["en_us", "de_de", "es_419"],
        help="List of language codes to process (e.g., en_us de_de fr_fr)"
    )
    
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=100,
        help="Maximum samples per language"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./embeddings",
        help="Directory to save extracted features"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Precision for stored hidden states (float16 halves file size)"
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for dataset loading"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip languages that already have output files"
    )

    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Determine compute device."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_model_and_processor(model_name: str, device: str):
    """Load Wav2Vec2 model and feature extractor."""
    logger.info(f"Loading model: {model_name}")
    
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded on {device}")
        return feature_extractor, model
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def extract_representations(
    audio_array: np.ndarray,
    sampling_rate: int,
    feature_extractor,
    model,
    device: str,
    store_dtype: str = "float32",
) -> List[np.ndarray]:
    """Extract hidden states from audio."""
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Pass attention_mask when the feature extractor provides one (the large
    # XLSR/XLS-R models do); required for correctness once batch size > 1.
    model_inputs = {"input_values": inputs["input_values"]}
    if "attention_mask" in inputs:
        model_inputs["attention_mask"] = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True)

    # Convert to numpy and move to CPU (optionally down-casting to save space)
    hidden_states = [h.cpu().numpy().astype(store_dtype) for h in outputs.hidden_states]

    return hidden_states


def process_language(
    lang: str,
    args,
    feature_extractor,
    model,
    device: str
) -> List[Dict[str, Any]]:
    """Process a single language from the FLEURS dataset."""
    logger.info(f"Processing language: {lang}")
    
    try:
        # Load dataset
        logger.info(f"Loading FLEURS dataset for {lang}...")
        dataset = load_dataset(
            "google/fleurs",
            lang,
            split=args.split,
            trust_remote_code=args.trust_remote_code
        )
        
        # Limit samples if needed
        if len(dataset) > args.max_samples:
            dataset = dataset.select(range(args.max_samples))
            
        logger.info(f"Processing {len(dataset)} samples")
        
        all_features = []
        
        for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {lang}")):
            try:
                audio = np.asarray(sample['audio']['array'], dtype=np.float32)
                sampling_rate = sample['audio']['sampling_rate']

                # wav2vec2 assumes 16 kHz and does not resample internally.
                # FLEURS is already 16 kHz; this guard keeps other datasets correct.
                if sampling_rate != TARGET_SR:
                    audio = librosa.resample(
                        audio, orig_sr=sampling_rate, target_sr=TARGET_SR
                    )
                    sampling_rate = TARGET_SR

                # Extract features
                hidden_states = extract_representations(
                    audio, sampling_rate, feature_extractor, model, device,
                    store_dtype=args.dtype,
                )

                feature_dict = {
                    # NOTE: FLEURS 'id' is the FLORES *sentence* id, not unique per
                    # recording — the same sentence can appear for multiple speakers.
                    'id': sample.get('id', idx),
                    'language': lang,
                    'transcription': sample.get('transcription', ''),
                    'raw_transcription': sample.get('raw_transcription', ''),
                    'hidden_states': hidden_states,  # List of layer arrays
                    'audio_length': len(audio) / sampling_rate,
                    'sampling_rate': sampling_rate
                }
                
                all_features.append(feature_dict)
                
            except Exception as e:
                logger.warning(f"Error on sample {idx} for {lang}: {e}")
                continue
                
        return all_features
        
    except Exception as e:
        logger.error(f"Failed to process language {lang}: {e}")
        return []


def save_features(features: List[Dict], output_path: str):
    """Save features to pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    
    logger.info(f"Saved {len(features)} samples to {output_path}")


def print_sample_info(features: List[Dict]):
    """Print diagnostic information about the extracted features."""
    if not features:
        return
        
    sample = features[0]
    print(f"\nSample structure:")
    print(f"  ID: {sample['id']}")
    print(f"  Language: {sample['language']}")
    print(f"  Transcription: {sample['transcription'][:50]}...")
    print(f"  Audio length: {sample['audio_length']:.2f}s")
    print(f"  Number of layers: {len(sample['hidden_states'])}")
    print(f"  Layer shapes:")
    for i, h in enumerate(sample['hidden_states']):
        print(f"    Layer {i}: {h.shape}")


def main():
    args = parse_args()
    
    # Setup
    device = setup_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Languages: {args.languages}")
    logger.info(f"  Max samples: {args.max_samples}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Device: {device}")
    
    # Load model once for all languages
    feature_extractor, model = load_model_and_processor(args.model, device)
    
    # Process each language
    for lang in args.languages:
        print(f"\n{'='*60}")
        print(f"Processing: {lang}")
        print(f"{'='*60}")
        
        output_file = os.path.join(args.output_dir, f"{lang}_features.pkl")
        
        # Skip if resuming and file exists
        if args.resume and os.path.exists(output_file):
            logger.info(f"Skipping {lang} (already exists)")
            continue
        
        # Process and save
        features = process_language(lang, args, feature_extractor, model, device)
        
        if features:
            save_features(features, output_file)
            print_sample_info(features)
        else:
            logger.warning(f"No features extracted for {lang}")
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()