import sys
import traceback

from mainapp import run_demo
from helloapp import HelloApp

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "hello":
        print("Running hello world application")
        sys.argv.pop(1)
        try:
            HelloApp().run()
        except Exception:
            print(traceback.format_exc())
    elif sys.argv[1] == "main":
        sys.argv.pop(1)
        try:
            print("Running main application")
            run_demo()
        except Exception:
            print(traceback.format_exc())
