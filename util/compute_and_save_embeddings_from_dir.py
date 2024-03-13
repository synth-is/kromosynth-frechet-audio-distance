# compute_and_save_embeddings_from_dir.py

# Command line utility (CLI) to compute and save embeddings from a directory of audio files.

# The CLI and the utility function provide a convenient way to compute and save embeddings from a directory of audio files using the Frechet Audio Distance library. This can be useful for preprocessing audio data for machine learning tasks such as audio classification, retrieval, and similarity analysis.

# The compute_and_save_embeddings_from_dir function takes the input directory, output directory, model name, sample rate, use_pca, use_activation, and verbose as arguments. It creates an instance of the FrechetAudioDistance class with the provided arguments and iterates over the audio files in the input directory. For each audio file, it computes the embeddings using the FrechetAudioDistance class and saves them as .npy files in the output directory.

# Usage:
# python -m util.compute_and_save_embeddings_from_dir --input_dir <input_dir> --output_dir <output_dir> --model_name <model_name> --sample_rate <sample_rate> --use_pca <use_pca> --use_activation <use_activation> --verbose <verbose> 

# Example:
# python -m util.compute_and_save_embeddings_from_dir --input_dir data --output_dir embeddings --model_name vggish --sample_rate 16000 --use_pca False --use_activation False --verbose False

# Arguments:
#   --input_dir: The directory containing the audio files.
#   --output_dir: The directory to save the embeddings.
#   --model_name: The name of the model to use. Options: vggish, pann.
#   --sample_rate: The sample rate of the audio files.
#   --use_pca: Whether to use PCA to reduce the dimensionality of the embeddings.
#   --use_activation: Whether to use the activation layer of the model.
#   --verbose: Whether to print the progress.

# The embeddings are saved in the output directory as .npy files with the same name as the audio files.

import argparse
import os
from frechet_audio_distance import FrechetAudioDistance
import numpy as np

def compute_and_save_embeddings_from_dir(input_dir, output_dir, model_name, sample_rate, use_pca, use_activation, verbose):
    """Computes and saves embeddings from a directory of audio files.
    Args:
        input_dir: The directory containing the audio files.
        output_dir: The directory to save the embeddings.
        model_name: The name of the model to use. Options: vggish, pann.
        sample_rate: The sample rate of the audio files.
        use_pca: Whether to use PCA to reduce the dimensionality of the embeddings.
        use_activation: Whether to use the activation layer of the model.
        verbose: Whether to print the progress.
    """
    print(f"model_name: {model_name}")
    print(f"sample_rate: {sample_rate}")
    print(f"use_pca: {use_pca}")
    print(f"use_activation: {use_activation}")
    frechet = FrechetAudioDistance(
        model_name=model_name,
        sample_rate=sample_rate,
        use_pca=use_pca,
        use_activation=use_activation,
        verbose=verbose
    )
    embeds_path = os.path.join(output_dir, "embeds.npy")
    frechet.extract_and_save_embeddings(input_dir, embeds_path)

def main():
    parser = argparse.ArgumentParser(description="Command line utility to compute and save embeddings from a directory of audio files.")
    parser.add_argument("--input_dir", type=str, required=True, help="The directory containing the audio files.")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to save the embeddings.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use. Options: vggish, pann.")
    parser.add_argument("--sample_rate", type=int, required=True, help="The sample rate of the audio files.")
    # https://stackoverflow.com/a/15008806/169858 :
    parser.add_argument("--use_pca", action=argparse.BooleanOptionalAction, help="Whether to use PCA to reduce the dimensionality of the embeddings.")
    parser.add_argument("--use_activation", action=argparse.BooleanOptionalAction, help="Whether to use the activation layer of the model.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, help="Whether to print the progress.")
    args = parser.parse_args()
    print(f"args: {args}")
    compute_and_save_embeddings_from_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        sample_rate=args.sample_rate,
        use_pca=args.use_pca,
        use_activation=args.use_activation,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
