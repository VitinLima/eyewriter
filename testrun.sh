source ./venv/bin/activate
cd EyeWriter/
buildozer android deploy run logcat | grep "python"
