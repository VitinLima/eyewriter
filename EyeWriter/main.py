import sys
import traceback

from kivy.utils import platform

def run_main():
    if platform == 'linux':
        #from maindesktop import run_desktop
        from mainandroid import run_android
        print("Running main application on desktop")
        run_android()

    elif platform == 'android':
        from mainandroid import run_android
        print("Running main application on android")
        run_android()

if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            print("Running default application")
            run_main()
            # from helloapp import HelloApp
            # HelloApp().run()
            #from mainandroid import run_android
            #run_android()

        elif sys.argv[1] == "hello":
            sys.argv.pop(1)
            print("Running hello world application")

            from helloapp import HelloApp
            HelloApp().run()

        elif sys.argv[1] == "main":
            sys.argv.pop(1)
            run_main()
    except Exception:
        print(traceback.format_exc())
