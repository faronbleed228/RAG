from argparse import Namespace, ArgumentParser


def main() -> None:
    pass


def arument_parser() -> Namespace:
    parser = ArgumentParser("RAG")
    parser.add_argument("--index", type=bool)
