#! /bin/bash

#conda install -c conda-forge sphinx sphinx-argparse sphinx_rtd_theme sphinx-jsonschema
#pip install sphinxcontrib.blockdiag

cwd=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}

if [ $2 == "True" ];
then
    rm -rf _build
    rm -rf _autosummary
    make clean html
fi

if [ $1 == "True" ];
then
    if [ ! -f "_build/html/index.html"  ];
    then
        make clean html
    fi
    if [ -f "_build/html/index.html"  ];
    then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            open _build/html/index.html
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            open _build/html/index.html
        elif [[ "$OSTYPE" == "cygwin" ]]; then
            open _build/html/index.html
        elif [[ "$OSTYPE" == "msys" ]]; then
            start _build/html/index.html
                # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
        elif [[ "$OSTYPE" == "win32" ]]; then
            start _build/html/index.html
        else
            echo "Unknown os type, $OSTYPE, contribute option to package docs/run.sh"
        fi
    fi
fi

cd ${cwd}
