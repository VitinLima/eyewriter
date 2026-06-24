import sys
import traceback

from kivy.utils import platform


if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            print("Running default application")
            # from helloapp import HelloApp
            # HelloApp().run()
            from mainandroid import run_android
            run_android()

        elif sys.argv[1] == "hello":
            sys.argv.pop(1)
            print("Running hello world application")

            from helloapp import HelloApp
            HelloApp().run()

        elif sys.argv[1] == "main":
            sys.argv.pop(1)
            from mainandroid import run_android
            from maindesktop import run_desktop

            if platform == 'linux':
                print("Running main application on desktop")
                run_android()  # run_desktop()

            elif platform == 'android':
                print("Running main application on android")
                run_android()
    except Exception:
        print(traceback.format_exc())
