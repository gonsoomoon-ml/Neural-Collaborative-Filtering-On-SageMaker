from __future__ import absolute_import

import argparse
import json
import sys
import traceback
import os

print("hello world")

def main():  # pragma: no cover
    print("###### Staring in run_pipeline by gs ############:")

    os.environ["VAR1"]="hello VAR1"

if __name__ == "__main__":
    main()
