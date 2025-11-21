"""Module for initializing notebooks."""
import pathlib
import sys

import dotenv
import nest_asyncio

_ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent
sys.path.append(_ROOT_PATH.resolve().as_posix())
dotenv.load_dotenv(dotenv.find_dotenv())
nest_asyncio.apply()
