
source ./venv/bin/activate
cd EyeWriter/
( rm ./buildlog.txt || true) && buildozer android debug | tee ./buildlog.txt
if [ 1 -eq 1 ] ; then
echo "Press Enter to continue..." ; read
fi
source ./venv/bin/activate ; cd EyeWriter/ ; buildozer android deploy run logcat | grep "python"
