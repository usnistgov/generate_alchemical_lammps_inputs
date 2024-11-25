<!--- GitHub Badges only --->
[![Build Status](https://github.com/usnistgov/generate_alchemical_lammps_inputs/workflows/CI/badge.svg)](https://github.com/usnistgov/generate_alchemical_lammps_inputs/actions?query=workflow%3ACI)
--->

### [NIST Disclaimer][nist-disclaimer]

Certain commercial equipment, instruments, or materials are identified in this paper to foster understanding. Such identification does not imply recommendation or endorsement by the National Institute of Standards and Technology, nor does it imply that the materials or equipment identified are necessarily the best available for the purpose.

Generate Alchemical LAMMPS Inputs
==============================

 Parsers for extracting alchemical data from LAMMPS output files.

> **IMPORTANT**
> Per the [GitHub ROB][gh-rob] and [NIST Suborder 1801.02][nist-s-1801-02],
your README should contain:
> 1. Software or Data Description
>    - Statements of purpose and maturity
>    - Description of the repository contents
>    - Technical installation instructions, including operating
>      system or software dependencies
> 1. Contact Information
>    - PI name, NIST OU, Division, and Group names
>    - Contact email address at NIST
> 1. Related Material
>    - URL for associated project on the NIST website or other Department
>      of Commerce page, if available
>    - References to user guides if stored outside of GitHub
> 1. Directions on appropriate citation with example text
> 1. References to any included non-public domain software modules,
>    and additional license language if needed, *e.g.* [BSD][li-bsd],
>    [GPL][li-gpl], or [MIT][li-mit]
>
> The more detailed your README, the more likely our colleagues
> around the world are to find it through a Web search. For general
> advice on writing a helpful README, please review
> [*Making Readmes Readable*][18f-guide] from 18F and Cornell's
> [*Guide to Writing README-style Metadata*][cornell-meta].

## Documentation
Checkout the documentation...:
 - locally: Run the following in the command line: ``python -m md_spa -d``
 - on [GitLab](https://jac16.ipages.nist.gov/generate_alchemical_lammps_inputs)
 - on GitHub (coming soon)

## Installation

* Step 1: Download the master branch from our gitlab page as a zip file, or clone it to your working directory with:

    GitLab: ``git clone https://gitlab.nist.gov/gitlab/jac16/generate_alchemical_lammps_inputs``
    
    GitHub: ``git clone https://github.com/usnistgov/generate_alchemical_lammps_inputs``

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

> **IMPORTANT**
> If your repository includes any software or data that is licensed by
> a third party, create a separate file for third-party licenses
> (`THIRD_PARTY_LICENSES.md` is recommended) and include copyright and
> licensing statements in compliance with the conditions of those licenses.

---

### Acknowledgements

Project based on the
[NIST Internal Cookiecutter](https://gitlab.nist.com/jaclark5/cookiecutter_template_nist_python) version 0.0.

<!-- References -->

[18f-guide]: https://github.com/18F/open-source-guide/blob/18f-pages/pages/making-readmes-readable.md
[cornell-meta]: https://data.research.cornell.edu/content/readme
[gh-rob]: https://odiwiki.nist.gov/pub/ODI/GitHub/GHROB.pdf
[li-bsd]: https://opensource.org/licenses/bsd-license
[li-gpl]: https://opensource.org/licenses/gpl-license
[li-mit]: https://opensource.org/licenses/mit-license
[nist-disclaimer]: https://www.nist.gov/open/license
[nist-s-1801-02]: https://inet.nist.gov/adlp/directives/review-data-intended-publication
[nist-open]: https://www.nist.gov/open/license#software
