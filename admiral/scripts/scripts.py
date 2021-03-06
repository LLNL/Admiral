#!/usr/bin/env python

import argparse
import os

from admiral.scripts import train_script as train
from admiral.scripts import analyze_script as analyze
from admiral.scripts import play_script as play
from admiral.scripts import make_runnable_script as runnable

EXAMPLE_USAGE = """
Example usage for training:
    admiral train my_experiment.py

Example usage for analysis:
    admiral analyze my_experiment_directory/ my_analysis_script.py

Example usage for playing:
    admiral play my_experiment_directory/ --some-args

Example usage for converting to runnable script:
    admiral make-runnable my_experiment.py --some-args
"""

def cli():
    parser = argparse.ArgumentParser(
        prog='admiral',
        description="Train, analyze, and play MARL policies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_USAGE
    )
    subparsers = parser.add_subparsers(dest='command')
    train_parser = train.create_parser(subparsers)
    analyze_parser = analyze.create_parser(subparsers)
    play_parser = play.create_parser(subparsers)
    runnable_parser = runnable.create_parser(subparsers)
    parameters = parser.parse_args()
    path_config = os.path.join(os.getcwd(), parameters.configuration)

    if parameters.command == 'train':
        train.run(path_config)
    elif parameters.command == 'analyze':
        full_path_subscript = os.path.join(os.getcwd(), parameters.subscript)
        analyze.run(path_config, full_path_subscript, parameters)
    elif parameters.command == 'play':
        play.run(path_config, parameters)
    elif parameters.command == 'make-runnable':
        runnable.run(path_config, parameters)
    else:
        parser.print_help()