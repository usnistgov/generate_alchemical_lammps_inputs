# Compiling Generate Alchemical LAMMPS Inputs's Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).

Reinstall the package with documentation dependencies. From the root directory of the package:
```bash
pip install .[doc]
```

Once installed, you can use the `Makefile` in this directory to compile static HTML pages by
```bash
make html
```

The compiled docs will be in the `_build` directory and can be viewed by opening `index.html` (which may itself
be inside a directory called `html/` depending on what version of Sphinx is installed).
