from argparse import ArgumentParser, Namespace
from transformers import (
    
)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="",
        default="agnews",
    )
    args = parser.parse_args()
    return args

def main(args):
    

    # training procedure
    # load data, define dataloader
    # define model
    # train
    # eval and test

    # attack procedure
    # load victim model
    # define attack recipe
    # do attack and compute result / generate adv sample


if __name__ == '__main__':
    args = parse_args()
    main(args)
