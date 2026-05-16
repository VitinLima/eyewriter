# conda init
# conda activate ./condaenv

# Adjust the paths!
export ANDROIDSDK="$HOME/android-app/"
export ANDROIDNDK="$HOME/android-app/android-ndk-r27d"
export ANDROIDAPI="27"  # Target API version of your application
export NDKAPI="24"  # Minimum supported API version of your application
export ANDROIDNDKVER="r10e"  # Version of the NDK you installed

# p4a apk --private ./App2.py --package=org.example.App2 --name "My application" --version 0.1 --bootstrap=sdl2 --requirements=python3,eyetrax,gTTS,pydub,numpy==v1.26.4,opencv-python --arch=arm64-v8a --arch=armeabi-v7a
# p4a apk --private . --package=org.example.app --name "My application" --version 0.1 --bootstrap=sdl2 --requirements=python3 --arch=arm64-v8a --arch=armeabi-v7a 
p4a apk --private . --package=org.example.myapp --use-setup-py --name "My application" --version 0.1 --bootstrap=sdl2 --requirements=python3,kivy --arch=arm64-v8a --arch=armeabi-v7a