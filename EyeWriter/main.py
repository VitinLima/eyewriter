import sys
import traceback

from kivy.utils import platform

from mainandroid import run_android
from maindesktop import run_desktop
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
        if platform == 'linux':
            try:
                print("Running main application on desktop")
                # run_desktop()
                run_android()
            except Exception:
                print(traceback.format_exc())
        elif platform == 'android':
            try:
                print("Running main application on android")
                run_android()
            except Exception:
                print(traceback.format_exc())
