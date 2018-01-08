# Sphinx Instructions

A collections of notes based on [this](http://gisellezeno.com/tutorials/sphinx-for-python-documentation.html), [readthedocs](https://github.com/rtfd/sphinx_rtd_theme) and [this](https://github.com/GaretJax/sphinx-autobuild).

## Installations

`sphinx` comes with Anaconda but not other packages it needs.

```
# read the docs theme
pip install sphinx_rtd_theme sphinx-autobuild
```

## Instructions

```
cd /git/project
mkdir docs
cd docs
sphinx-quickstart
```

Key questions:

* when it asks to separate the `source` and `build` directories, type **y** and press enter
* when it asks for the `autodoc` extension, type **y** and press enter

```
cd source
nano conf.py
```

Uncomment and change the following:

```
sys.path.insert(0, os.path.abspath('../..'))

import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
# following line is not needed for > 0.2.5
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
```

Run command:

```
cd ..
sphinx-apidoc -f -o source/ ../mypackage/
make html
```
