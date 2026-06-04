from pythonforandroid.recipe import PyProjectRecipe

class EyeTraxRecipe(PyProjectRecipe):
    site_packages_name = "eyetrax"
    version = 'master'
    url = 'https://github.com/ck-zhang/EyeTrax/archive/refs/heads/master.zip'
    depends = ["setuptools"]
    patches = ["eyetrax_patch.patch"]


recipe = EyeTraxRecipe()
