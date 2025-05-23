import argparse
from pathlib import Path
from audio_utils import add_noise

def main():
    """
    Add noise to the audio file.
    """
    argparser = argparse.ArgumentParser(
        description="Add noise to an audio file."
    )
    argparser.add_argument(
        "input_file",
        type=str,
        help="Path to the input audio file."
    )
    argparser.add_argument(
        "output_file",
        type=str,
        help="Path to the output audio file with added noise."
    )
    argparser.add_argument(
        "--noise_level",
        "-n",
        type=float,
        default=0.05,
        help="Noise level to be added to the audio file."
    )
    argparser.add_argument(
        "--noise_type",
        type=str,
        choices=["white", "pink", "brown"],
        default="white",
        help="Type of noise to be added to the audio file."
    )
    
    args = argparser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    add_noise(
        input_audio_path=input_file,
        output_noisy_audio_path=output_file,
        noise_level=args.noise_level,
        noise_type=args.noise_type
    )

if __name__ == "__main__":
    main()