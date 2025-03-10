<!--- GitHub Badges only --->
[![Build Status](https://github.com/usnistgov/generate_alchemical_lammps_inputs/workflows/CI/badge.svg)](https://github.com/usnistgov/generate_alchemical_lammps_inputs/actions?query=workflow%3ACI)

### [NIST Disclaimer][nist-disclaimer]

Certain commercial equipment, instruments, or materials are identified in this paper to foster understanding. Such identification does not imply recommendation or endorsement by the National Institute of Standards and Technology, nor does it imply that the materials or equipment identified are necessarily the best available for the purpose.

Generate Alchemical LAMMPS Inputs
==============================

Functions to generate LAMMPS inputs for alchemical calculations. This package was produced for generating the simulation input file for the completion of the publication "Hydration Contribution to the Solvation Free Energy of Water-Soluble Polymers". These functions include inputs used for thermodynamic integration, Bennett Acceptance Ratio (BAR), or Multi-state Bennett Acceptance Ratio (MBAR).

## Documentation

 - locally: Run the following in the command line: ``python -m generate_alchemical_lammps_inputs -d``
 - online: [GitHub][docs4nist]

## Installation

* Step 1: Download the master branch from our github page as a zip file, or clone it to your working directory with:

    ``git clone https://github.com/usnistgov/generate_alchemical_lammps_inputs``

* Step 2 (Optional): If you are using conda and you want to create a new environment for this package you may install with:

    ``conda env create -f requirements.yaml``

* Step 3: Install package with:

    ``pip install generate_alchemical_lammps_inputs/.``

    or change directories and run

    ``pip install .``

    Adding the flag ``-e`` will allow you to make changes that will be functional without reinstallation.

* Step 4: Initialize pre-commits (for developers)

    ``pre-commit install``

## LICENSE

The license in this repository is superseded by the most updated language
on of the Public Access to NIST Research [*Copyright, Fair Use, and Licensing Statement for SRD, Data, and Software*][nist-open].

## Contact

Jennifer A. Clark, PhD\
[Debra J. Audus, PhD][daudus] (debra.audus@nist.gov)\
[Jack F. Douglas, PhD][jdouglas]

## Affiliation
[Polymer Analytics Project][polyanal]\
[Polymer and Complex Fluids Group][group1]\
[Materials Science and Engineering Division][msed]\
[Material Measurement Laboratory][mml]\
[National Institute of Standards and Technology][nist]

## Citation

Clark, J. A.; Audus, D. J.; Douglas, J. F. Python Package for Generating LAMMPS Input Scripts for Alchemical Processes: Generate_alchemical_lammps_inputs, 2024. https://doi.org/10.18434/mds2-3641.

<!-- References -->

[18f-guide]: https://github.com/18F/open-source-guide/blob/18f-pages/pages/making-readmes-readable.md
[cornell-meta]: https://data.research.cornell.edu/content/readme
[gh-rob]: https://odiwiki.nist.gov/pub/ODI/GitHub/GHROB.pdf
[li-bsd]: https://opensource.org/licenses/bsd-license
[li-gpl]: https://opensource.org/licenses/gpl-license
[li-mit]: https://opensource.org/licenses/mit-license
[nist-disclaimer]: https://www.nist.gov/open/license
[nist-open]: https://www.nist.gov/open/license#software
[docs4nist]: https://www.nist.gov/docs4nist/
[daudus]: https://www.nist.gov/people/debra-audus
[jdouglas]: https://www.nist.gov/people/jack-f-douglas
[polyanal]: https://www.nist.gov/programs-projects/polymer-analytics
[group1]: https://www.nist.gov/mml/materials-science-and-engineering-division/polymers-and-complex-fluids-group
[msed]: https://www.nist.gov/mml/materials-science-and-engineering-division
[mml]: https://www.nist.gov/mml
[nist]: https://www.nist.gov
