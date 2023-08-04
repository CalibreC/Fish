import argparse


def make_parser():
    parser = argparse.ArgumentParser("Genshin Fishing")
    parser.add_argument("--mode", default="genshin", type=str, help="train or test")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    print(args.mode)
    pass
