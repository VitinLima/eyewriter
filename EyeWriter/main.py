import sys
import traceback

from mainapp import run_demo
from helloapp import App as HelloApp

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "main":
        try:
            run_demo()
        except Exception:
            print(traceback.format_exc())
    elif sys.argv[1] == "test":
        try:
            HelloApp().run()
        except Exception:
            print(traceback.format_exc())
