from pythonforandroid.recipe import PyProjectRecipe

class EyeTraxRecipe(PyProjectRecipe):
    site_packages_name = "eyetrax"
    version = 'master'
    url = 'https://github.com/ck-zhang/EyeTrax/archive/refs/heads/master.zip'
    depends = ["setuptools", "mediapipe>=0.10",
               "numpy>=1.22,<2",
            #    "scikit-learn>=1.3",
            #    "scipy>=1.10",
            #    "screeninfo>=0.8",
            #    "pyvirtualcam>=0.10"
               ]
    patches = ["eyetrax_patch.patch"]


recipe = EyeTraxRecipe()
