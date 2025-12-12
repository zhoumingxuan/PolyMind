import os
import sys

sys.path.append(os.path.dirname(__file__))

from search_service import web_search
from config import ConfigHelper

config = ConfigHelper()

def run_search(query: str, time_filter: str = "none"):
    return web_search(query)


if __name__ == "__main__":
    query_arg = sys.argv[1] if len(sys.argv) > 1 else "示例查询"
    time_arg = sys.argv[2] if len(sys.argv) > 2 else "none"

    try:
        run_search(query_arg, time_arg)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Unexpected error: {exc}")
