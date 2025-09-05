from huggingface_hub import snapshot_download
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face Hub")
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Local directory to store downloaded files"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    snapshot_download(
        repo_id="nvidia/PhysicalAI-SmartSpaces",  #
        repo_type="dataset",                      #
        allow_patterns="MTMC_Tracking_2025/*",   #
        ignore_patterns=".",            #
        local_dir=args.data_root
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
