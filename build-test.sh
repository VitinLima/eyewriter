source ./venv/bin/activate
cd EyeWriter/
clear
( rm ./buildlog.txt || true )
buildozer android debug | tee ./buildlog.txt
buildozer android deploy run logcat | grep "python"
