
source ./venv/bin/activate
cd EyeWriter/
rm -f ./buildlog.txt
if [[ 0 -eq 1 ]] ; then
opt_cleanmake="clean"
fi
buildozer android $opt_cleanmake debug | tee ./buildlog.txt
if [ 1 -eq 1 ] ; then
echo "Press Enter to continue..." ; read
fi
{
buildozer android deploy run logcat | grep "python"
}
