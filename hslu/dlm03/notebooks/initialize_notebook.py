"""Module for initializing notebooks."""
import logging
import pathlib
import sys

import dotenv
import nest_asyncio

logging.disable(logging.WARNING)

_ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent
sys.path.append(_ROOT_PATH.resolve().as_posix())
dotenv.load_dotenv(dotenv.find_dotenv())
nest_asyncio.apply()
