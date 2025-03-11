"""Functions to generate LAMMPS inputs for alchemical calculations.

For clarity, we would like to distinguish the difference between :math:`\lambda` and :math:`\lambda'` (or :math:`\lambda_2`). We refer to :math:`\lambda` as
the scaling factor on the potential energy of the equilibrated system, so that when this value is changed, the system undergoes another equilibration
step. On the other hand, :math:`\lambda'` is the value used to scale the potentials despite those the configurations being equilibrated
for a different value of :math:`\lambda`. The value of :math:`\lambda'` is used in two instances. First, in thermodynamic integration (TI), values of :math:`\lambda'`
that are very close to :math:`\lambda` can be used to calculate the derivative of the free energy with respect to :math:`\lambda`. This is
needed because LAMMPS does not compute explicit derivatives, although one should check whether they can derive an explicit expression, they cannot for changes of
:math:`\lambda'` in the soft Lennard-Jones (LJ) potential.

Because generating LAMMPS input files can be cumbersome, functions have been included to generate the appropriate sections. If a linear approximation
can be made to calculate :math:`U_{\lambda,\lambda'}` from :math:`U_{\lambda}` in post-processing, we recommend using
:func:`generate_input_linear_approximation()`. If a linear approximation cannot be made (such as changing
:math:`\lambda` in the soft-LJ potential) we recommend running a loop over all values of :math:`\lambda` saving frames spaced to be
independent samples, and an output file with small perturbations in :math:`\lambda'` to calculate the derivative for TI in
post-processing. This is achieved with :func:`generate_traj_input()`. After this first simulation, we then
recommend the files needed for MBAR are generated using the `rerun <https://docs.lammps.org/rerun.html>`_ feature in LAMMPS.
Breaking up the computation like this will allow one to add additional points to their MBAR analysis without repeating the
points from an initial simulation. Generating the file for a rerun is achieved with
:func:`generate_rerun_mbar()`.

Notice that the output files do not contain the header information expected
in LAMMPS as that is system specific and left to the user.

Note that in LAMMPS, `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_ changes values of :math:`\lambda` and
`compute fep <https://docs.lammps.org/compute_fep.html>`_ changes values of :math:`\lambda'`.

"""

import numpy as np


def _check_fix_adapt_changes_format(fix_adapt_changes):
    """Check that the input list describes a valid LAMMPS input

    Parameters
    ----------
    fix_adapt_changes : list
        A list of lists containing the following information from `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_:
        attribute (str), args (tuple). This option is useful when a parameter that has been varied and is set to another value
        in this simulation, e.g., lambda when the Coulomb potential is set to zero. Using this feature avoids complications
        with writing the pair potential information in the data file. The attributes supported in LAMMPS are:

        - pair; arguments are: pair_style parameter solvent_type(s) solute_type(s)

        - pair_style (str) : String of LAMMPS pair style being changes
        - parameter (str) : Parameter being varied, see table in `compute fep <https://docs.lammps.org/compute_fep.html>`_
          for the options in your pair-potential
        - solvent_type (str) : String defining atom types in the solvent (no spaces), lists are denoted with an asterisk
        - solute_type (str) : String defining atom types in the solute (no spaces), lists are denoted with an asterisk

        - atom; arguments are: parameter atom_type(s). These lines will be scaled by the variation of a "lambda" parameter
            whose range is defined in `parameter_range`, where the starting and end points are multiplied by ``variable_initial``.

        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
        - atom_type (str) : String defining atom types being affected, lists are denoted with an asterisk
        - variable_initial (float) : Specify the initial value of the variable

        - kspace; arguments are: None

    """

    if not isinstance(fix_adapt_changes, list):
        raise TypeError("fix_adapt_changes is expected to be a list.")
    if not all([len(x) == 2 for x in fix_adapt_changes]):
        raise ValueError(
            "fix_adapt_changes is expected to be to be a list of iterables of length two, containing"
            " an attribute defined by a str and an iterable containing the appropriate keywords."
        )

    for attribute, args in fix_adapt_changes:
        if attribute == "kspace":
            if len(args) != 0:
                raise ValueError("{}: {}\nNo arguments should be given for attribute: kspace".format(attribute, args))
        elif attribute == "pair":
            if len(args) != 4:
                raise ValueError(
                    "{}: {}\nFour arguments should be given for attribute: pair; pair_style, parameter, solvent_type, and solute_type".format(
                        attribute, args
                    )
                )
        elif attribute == "atom":
            if len(args) != 3:
                raise ValueError(
                    "{}: {}\nThree arguments should be given for attribute: atom;  atom_type, variable_initial".format(
                        attribute, args
                    )
                )


def generate_input_linear_approximation(
    parameter_range,
    parameter_change,
    fix_adapt_changes,
    temperature,
    types_solute,
    types_solvent,
    n_run_equil_steps=1000000,
    n_run_prod_steps=1000000,
    output_frequency=1000,
    output_file=None,
    fix_adapt_changes2=None,
    parameter2_value=None,
    is_charged=True,
    parameter_array=None,
):
    """Outputs the section of a LAMMPS input file that separates the Coulomb, nonbonded, and bond/angle/torsional contributions
    of the solute and solvent. As long as the parameter being changed is linearly dependent on the potential energy, these files for
    each value of the parameter can be used for thermodynamic integration (TI) or multi-state Bennett acceptance ratio (MBAR).

    The input data file for this script should be an equilibrated frame in the NPT ensemble. Notice that the input file contains
    the following keywords that you might replace with the values for your simulation using `sed`: TEMP, PRESS

    Parameters
    ----------
    parameter_range : list[float]
        Range of parameter values to be changed where the first value should be the value with which the system has been
        equilibrated.
    parameter_change : float
        The size of the step between parameter values, where a positive value increases and a negative value decreases.
        Take care that number of points needed to traverse the given range should result in an integer, otherwise
        LAMMPS will not end at the desired value.
    fix_adapt_changes : list
        A list of lists containing the following information from `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_:
        attribute (str), args (tuple), e.g., [["pair", ("lj/cut/soft", "lambda", 1, 2)]]. The attributes supported in LAMMPS are:

        - pair; arguments are: pair_style parameter solvent_type(s) solute_type(s)

        - pair_style (str) : String of LAMMPS pair style being changes
        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
        - solvent_type (str) : String defining atom types in the solvent (no spaces), lists are denoted with an asterisk
        - solute_type (str) : String defining atom types in the solute (no spaces), lists are denoted with an asterisk

        - atom; arguments are: parameter atom_type(s). These lines will be scaled by the variation of a "lambda" parameter
            whose range is defined in `parameter_range`, where the starting and end points are multiplied by ``variable_initial``.

        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
        - atom_type (str) : String defining atom types being affected, lists are denoted with an asterisk
        - variable_initial (float) : Specify the initial value of the variable

    temperature : float
        Temperature of the simulation. This will create a variable that is used in the FEP computes.
    types_solvent : str
        String defining all atom types in the solvent (no spaces) with lists defined using colons
    types_solute : str
        String defining all atom types in the solute (no spaces) with lists defined using colons
    n_run_equil_steps : int, default=1000000
        Number of steps in each equilibration run, first with a ramp from the old lambda value to the new value and then for
        equilibration at the new value. The total number of time steps taken per step in lambda (i.e., window) is
        2*n_run_equil_steps + n_run_prod_steps
    n_run_prod_steps : int, default=1000000
        Number of steps in each production run. The total number of time steps taken per step in lambda (i.e., window) is
        2*n_run_equil_steps + n_run_prod_steps
    output_frequency : int, default=1000
        Number of steps between thermo output and dump output.
    output_file : str, default=None
        File name and path for optional output file
    fix_adapt_changes2 : list, default=[]
        A list of lists containing the following information from `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_:
        attribute (str), args (tuple). This option is useful when a parameter that has been varied and is set to another value
        in this simulation, e.g., lambda when the Coulomb potential is set to zero. Using this feature avoids complications
        with writing the pair potential information in the data file. The attributes supported in LAMMPS are:

        - pair; arguments are: pair_style parameter solvent_type(s) solute_type(s)

        - pair_style (str) : String of LAMMPS pair style being changes
        - parameter (str) : Parameter being varied, see table in `compute fep <https://docs.lammps.org/compute_fep.html>`_
          for the options in your pair-potential
        - solvent_type (str) : String defining atom types in the solvent (no spaces), lists are denoted with an asterisk
        - solute_type (str) : String defining atom types in the solute (no spaces), lists are denoted with an asterisk

        - atom; arguments are: parameter atom_type(s). These lines will be scaled by the variation of a "lambda" parameter
            whose range is defined in `parameter_range`, where the starting and end points are multiplied by ``variable_initial``.

        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options such as "charge"
        - atom_type (str) : String defining atom types being affected, lists are denoted with an asterisk
        - variable_initial (float) : Specify the initial value of the variable

    parameter2_value : str, default=None
        Value to set ``parameter2`` in ``fix_adapt_changes2``.
    is_charged : bool, default=True
        This will add kspace components to the record of free energy changes
    parameter_array : list, default=None
        If not ``None``, this argument will overwrite the ``parameter_change`` argument and use the specified steps in
        the parameter. Note that the first value should represent the state that the trajectory is equilibrated at.

    Returns
    -------
    file : list[str]
        List of strings representing lines in a file

    """
    if parameter_array is None:
        prec = len(repr(parameter_change).split(".")[-1])
        parameter_array = np.concatenate(
            (np.arange(parameter_range[0], parameter_range[1], parameter_change), [parameter_range[1]])
        )
        flag_array = False
    else:
        parameter_array = np.array(parameter_array)
        prec = int(np.max([len(repr(x).split(".")[-1]) for x in parameter_array]))
        flag_array = True

    nblocks = len(parameter_array)
    if nblocks < 2:
        if flag_array:
            raise ValueError(
                "If `parameter_array is not None`, an array of parameters must be provided, instead of {}".format(
                    parameter_array
                )
            )
        else:
            raise ValueError(
                "The argument `parameter_change` must be appropriately positive or negative to span the `parameter_range`"
            )

    if parameter2_value is None and fix_adapt_changes2:
        raise ValueError("If fix_adapt_changes2 is provided, so must parameter2_value")
    if parameter2_value is not None and not fix_adapt_changes2:
        raise ValueError("If parameter2_value is provided, so must fix_adapt_changes2")

    _check_fix_adapt_changes_format(fix_adapt_changes)
    if fix_adapt_changes[0][0] == "pair":
        name1 = "-".join([fix_adapt_changes[0][1][0].replace("/", "-"), fix_adapt_changes[0][1][1]])
    elif fix_adapt_changes[0][0] == "atom":
        name1 = str(fix_adapt_changes[0][1][0])
    else:
        name1 = "kspace"
    if fix_adapt_changes2:
        _check_fix_adapt_changes_format(fix_adapt_changes2)
        if fix_adapt_changes2[0][0] == "pair":
            name2 = "-".join([fix_adapt_changes2[0][1][0].replace("/", "-"), fix_adapt_changes2[0][1][1]])
        elif fix_adapt_changes2[0][0] == "atom":
            name2 = "-".join([str(fix_adapt_changes2[0][1][0]), str(fix_adapt_changes2[0][1][-1] * parameter2_value)])
        else:
            name2 = "kspace"

    file = [
        "\n# Variables and System Conditions\n",
        f"variable TK equal {temperature}\n",
        f"variable freq equal {output_frequency}\n",
        f"variable runtime_equil equal {n_run_equil_steps}\n",
        f"variable runtime_prod equal {n_run_prod_steps}\n",
        "variable pinst equal press\n",
        "variable tinst equal temp\n",
        "variable vinst equal vol\n",
        "variable pe equal pe\n",
        f"variable nblocks equal {nblocks} \n",
        f"variable lambdas vector [{','.join([str(np.round(x, decimals=prec)) for x in parameter_array])}]\n",
        "\n",
        "thermo ${freq}\n",
        "\n# Group atoms\n",
        f"group solute type {types_solute}\n",
        f"group solvent type {types_solvent}\n",
        "\n# Set-up Loop\n",
        "variable runid loop 1 ${nblocks} pad\n",
        "    label runloop1\n",
        "\n    # Adjust param for the box and equilibrate\n",
        "    variable param equal v_lambdas[v_runid]\n",
        '    if "${runid} == 1" then &\n',
        '        "jump SELF skipequil"\n',
        "    variable ind equal v_runid-1\n",
        "    variable param0 equal v_lambdas[v_ind]\n",
        "    variable paramramp equal ramp(v_param0,v_param)\n",
    ]
    ind = 9
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            file[ind + 2 * ii : ind + 2 * ii] = [f"variable deltacdmA{ii} equal {args[-1]}*v_deltacdm\n"]
            tmp = str(args[-1])
            if tmp[0] == "-":
                file[ind + 2 * ii + 1 : ind + 2 * ii + 1] = [f"variable deltacdm2A{ii} equal {tmp[1:]}*v_deltacdm\n"]
            else:
                file[ind + 2 * ii + 1 : ind + 2 * ii + 1] = [f"variable deltacdm2A{ii} equal -{tmp}*v_deltacdm\n"]

    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            file.append(f"    variable paramrampA{ii} equal {args[-1]}*ramp(v_param0,v_param)\n")
            file.append(f"    variable paramA{ii} equal {args[-1]}*v_param\n")
    # Add previously changed parameter if it exists
    if fix_adapt_changes2:
        file[ind:ind] = [
            "\n# Set Previous Change\n",
            f"variable param2 equal {parameter2_value}\n",
        ]
        ind += 2
        for ii, (attribute, args) in enumerate(fix_adapt_changes2):
            if attribute == "atom":
                file[ind + ii : ind + ii] = [f"variable param2A{ii} equal {args[-1]}*v_param2\n"]
        ind += ii + 1
        file[ind:ind] = ["fix ADAPT2 all adapt/fep 1 &\n"]
        ind += 1
        for ii, (attribute, args) in enumerate(fix_adapt_changes2):
            if attribute == "atom":
                tmp = " ".join([str(x) for x in args[:-1]])
                file[ind + ii : ind + ii] = [f"    {attribute} {tmp}  v_param2A{ii} &\n"]
            else:
                if attribute == "pair":
                    arg1, arg2 = args[2:]
                    arg1 = str(arg1).replace("*", " ").split(" ")
                    arg2 = str(arg2).replace("*", " ").split(" ")
                    if int(arg2[0]) < int(arg1[0]):
                        args = (args[0], args[1], args[3], args[2])
                tmp = " ".join([str(x) for x in args])
                file[ind + ii : ind + ii] = [f"    {attribute} {tmp}  v_param2 &\n"]
        file[ind + ii] = file[ind + ii].replace("&", "")

    file.append("    fix ADAPT all adapt/fep 1 &\n")
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"        {attribute} {tmp}  v_paramrampA{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1 = str(arg1).replace("*", " ").split(" ")
                arg2 = str(arg2).replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"        {attribute} {tmp}  v_paramramp &\n")
    file[-1] = file[-1].replace("&", "")
    file.extend(
        [
            "    thermo_style custom step v_paramramp "
            + " ".join(
                [f"v_paramrampA{ii}" for ii, (attribute, _) in enumerate(fix_adapt_changes) if attribute == "atom"]
            )
            + " temp press pe evdwl enthalpy\n",
            "    run ${runtime_equil} # Run Ramp\n",
            "    unfix ADAPT\n",
            "    fix ADAPT all adapt/fep ${freq} &\n",
        ]
    )
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"        {attribute} {tmp}  v_paramA{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1.replace("*", " ").split(" ")
                arg2.replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"        {attribute} {tmp}  v_param &\n")
    file[-1] = file[-1].replace("&", "")
    file.extend(
        [
            "    thermo_style custom step v_param "
            + " ".join([f"v_paramA{ii}" for ii, (attribute, _) in enumerate(fix_adapt_changes) if attribute == "atom"])
            + " temp press pe evdwl enthalpy\n",
            "    run ${runtime_equil} # Run Equil\n",
            "\n    label skipequil\n\n",
        ]
    )
    if fix_adapt_changes2:
        file.append(f"    write_data files/{name1}_" + "${param}_" + f"{name2}_{parameter2_value}.data\n")
    else:
        file.append(f"    write_data files/{name1}_" + "${param}.data\n")
    file.extend(
        [
            "\n    # Initialize computes\n",
            "    thermo_style custom step v_param temp press pe evdwl enthalpy\n",
            "    ## Compute PE for contributions for bonds, angles, dihedrals, and impropers\n",
            "    compute pe_solute_1 solute pe/atom bond angle dihedral improper # PE from nonpair/noncharged intramolecular interactions\n",
            "    compute pe_solvent_1 solvent pe/atom bond angle dihedral improper # PE from nonpair/noncharged intramolecular interactions\n",
            "    compute pe_solute_bond solute reduce sum c_pe_solute_1\n",
            "    compute pe_solvent_bond solvent reduce sum c_pe_solvent_1\n",
            "    ## Compute PE for contributions for pair (Includes real space component of coul)\n",
            "    compute pe_solute_pair solute group/group solute pair yes kspace no\n",
            "    compute pe_solvent_pair solvent group/group solvent pair yes kspace no\n",
            "    compute pe_inter_pair solute group/group solvent pair yes kspace no\n",
        ]
    )
    if is_charged:
        file.extend(
            [
                "    ## Compute PE for contributions for kspace component \n",
                "    compute pe_solute_kspace solute group/group solute pair no kspace yes\n",
                "    compute pe_solvent_kspace solvent group/group solvent pair no kspace yes\n",
                "    compute pe_inter_kspace solute group/group solvent pair no kspace yes\n",
                "    thermo_style custom step v_param temp press pe evdwl enthalpy &\n",
                "        c_pe_solute_bond c_pe_solute_pair c_pe_solute_kspace &\n",
                "        c_pe_solvent_bond c_pe_solvent_pair c_pe_solvent_kspace &\n",
                "        c_pe_inter_pair c_pe_inter_kspace\n",
            ]
        )
    else:
        file.extend(
            [
                "    thermo_style custom step v_param temp press pe evdwl enthalpy &\n",
                "        c_pe_solute_bond c_pe_solute_pair &\n",
                "        c_pe_solvent_bond c_pe_solvent_pair &\n",
                "        c_pe_inter_pair\n",
            ]
        )
    if fix_adapt_changes2:
        file.append("    fix FEPout all ave/time ${freq} 1 ${freq} v_param v_param2 v_tinst v_pinst v_vinst v_pe &\n")
    else:
        file.append("    fix FEPout all ave/time ${freq} 1 ${freq} v_param v_tinst v_pinst v_vinst v_pe &\n")
    if is_charged:
        file.extend(
            [
                "        c_pe_solute_bond c_pe_solute_pair c_pe_solute_kspace &\n",
                "        c_pe_solvent_bond c_pe_solvent_pair c_pe_solvent_kspace &\n",
                "        c_pe_inter_pair c_pe_inter_kspace &\n",
            ]
        )
    else:
        file.extend(
            [
                "        c_pe_solute_bond c_pe_solute_pair &\n",
                "        c_pe_solvent_bond c_pe_solvent_pair &\n",
                "        c_pe_inter_pair &\n",
            ]
        )
    if fix_adapt_changes2:
        file.append(f"        file files/linear_{name1}_" + "${param}_" + f"{name2}_{parameter2_value}.txt\n\n")
    else:
        file.append(f"        file files/linear_{name1}_" + "${param}.txt\n")
    file.extend(
        [
            "\n    run ${runtime_prod}\n\n",
            "    uncompute pe_solute_bond\n",
            "    uncompute pe_solute_1\n",
            "    uncompute pe_solvent_bond\n",
            "    uncompute pe_solvent_1\n",
            "    uncompute pe_solute_pair\n",
            "    uncompute pe_solvent_pair\n",
            "    uncompute pe_inter_pair\n",
        ]
    )
    if is_charged:
        file.extend(
            [
                "    uncompute pe_solute_kspace\n",
                "    uncompute pe_solvent_kspace\n",
                "    uncompute pe_inter_kspace\n",
            ]
        )
    file.extend(
        [
            '    if "${runid} != 1" then &\n',
            '        "unfix ADAPT"\n',
            "    unfix FEPout\n",
            "\n    next runid\n",
            "    jump SELF runloop1\n",
            "thermo_style custom v_param temp press pe evdwl enthalpy\n",
            "write_data final.data nocoeff\n",
        ]
    )
    if fix_adapt_changes2:
        file.extend(
            [
                "unfix ADAPT2\n",
            ]
        )

    if output_file is not None:
        with open(output_file, "w") as f:
            for line in file:
                f.write(line)

    return file


def generate_traj_input(
    parameter_range,
    parameter_change,
    fix_adapt_changes,
    temperature,
    n_run_equil_steps=1000000,
    n_run_prod_steps=1000000,
    output_frequency=1000,
    del_parameter=0.01,
    output_file=None,
    fix_adapt_changes2=None,
    parameter2_value=None,
    parameter_array=None,
):
    """Outputs the section of a LAMMPS input file that loops over the values of parameter being changed (e.g., lambda)
    Small perturbations in the potential energy are also output so that the derivative can be calculated for thermodynamic
    integration. Trajectories are produces so that files for MBAR analysis may be generated in post-processing.

    The input data file for this script should be an equilibrated frame in the NPT ensemble. Notice that the input file contains
    the following keywords that you might replace with the values for your simulation using `sed`: TEMP, PRESS

    Parameters
    ----------
    parameter_range : list[float]
        Range of parameter values to be changed where the first value should be the value with which the system has been
        equilibrated.
    parameter_change : float
        The size of the step between parameter values, where a positive value increases and a negative value decreases.
        Take care that number of points needed to traverse the given range should result in an integer, otherwise
        LAMMPS will not end at the desired value.
    fix_adapt_changes : list
        A list of lists containing the following information from `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_:
        attribute (str), args (tuple), e.g., [["pair", ("lj/cut/soft", "lambda", 1, 2)]]. The attributes supported in LAMMPS are:

        - pair; arguments are: pair_style parameter solvent_type(s) solute_type(s)

        - pair_style (str) : String of LAMMPS pair style being changes
        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
        - solvent_type (str) : String defining atom types in the solvent (no spaces), lists are denoted with an asterisk
        - solute_type (str) : String defining atom types in the solute (no spaces), lists are denoted with an asterisk

        - atom; arguments are: parameter atom_type(s). These lines will be scaled by the variation of a "lambda" parameter
            whose range is defined in `parameter_range`, where the starting and end points are multiplied by ``variable_initial``.

        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
        - atom_type (str) : String defining atom types being affected, lists are denoted with an asterisk
        - variable_initial (float) : Specify the initial value of the variable

    temperature : float
        Temperature of the simulation. This will create a variable that is used in the FEP computes.
    n_run_equil_steps : int, default=1000000
        Number of steps in each equilibration run, first with a ramp from the old lambda value to the new value and then for
        equilibration at the new value. The total number of time steps taken per step in lambda (i.e., window) is
        2*n_run_equil_steps + n_run_prod_steps
    n_run_prod_steps : int, default=1000000
        Number of steps in each production run. The total number of time steps taken per step in lambda (i.e., window) is
        2*n_run_equil_steps + n_run_prod_steps
    output_frequency : int, default=1000
        Number of steps between thermo output and dump output.
    del_parameter : float, default=0.01
        Change used to calculate the forward and backward difference used to compute the derivative through a central difference
        approximation.
    output_file : str, default=None
        File name and path for optional output file
    fix_adapt_changes2 : list, default=None
        A list of lists containing the following information from `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_:
        attribute (str), args (tuple). This option is useful when a parameter that has been varied and is set to another value
        in this simulation, e.g., lambda when the Coulomb potential is set to zero. Using this feature avoids complications
        with writing the pair potential information in the data file. The attributes supported in LAMMPS are:

        - pair; arguments are: pair_style parameter solvent_type(s) solute_type(s)

        - pair_style (str) : String of LAMMPS pair style being changes
        - parameter (str) : Parameter being varied, see table in `compute fep <https://docs.lammps.org/compute_fep.html>`_
          for the options in your pair-potential
        - solvent_type (str) : String defining atom types in the solvent (no spaces), lists are denoted with an asterisk
        - solute_type (str) : String defining atom types in the solute (no spaces), lists are denoted with an asterisk

        - atom; arguments are: parameter atom_type(s). These lines will be scaled by the variation of a "lambda" parameter
            whose range is defined in `parameter_range`, where the starting and end points are multiplied by ``variable_initial``.

        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options such as "charge"
        - atom_type (str) : String defining atom types being affected, lists are denoted with an asterisk
        - variable_initial (float) : Specify the initial value of the variable

    parameter2_value : float, default=None
        Value to set ``parameter2`` in ``fix_adapt_changes2``. Parameter that has been varied and is set to another value in this simulation, e.g., lambda when the Coulomb potential
        is set to zero. Using this feature avoids complications with writing the pair potential information in the data file.
        See table in `compute fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
    parameter_array : list, default=None
        If not ``None``, this argument will overwrite the ``parameter_change`` argument and use the specified steps in
        the parameter. Note that the first value should represent the state that the trajectory is equilibrated at.

    Returns
    -------
    file : list[str]
        List of strings representing lines in a file

    """
    del_parameter = np.abs(del_parameter)

    if parameter_array is None:
        prec = len(repr(parameter_change).split(".")[-1])
        parameter_array = np.concatenate(
            (np.arange(parameter_range[0], parameter_range[1], parameter_change), [parameter_range[1]])
        )
        flag_array = False
    else:
        parameter_array = np.array(parameter_array)
        prec = int(np.max([len(repr(x).split(".")[-1]) for x in parameter_array]))
        flag_array = True

    nblocks = len(parameter_array)
    if nblocks < 2:
        if flag_array:
            raise ValueError(
                "If `parameter_array is not None`, an array of parameters must be provided, instead of {}".format(
                    parameter_array
                )
            )
        else:
            raise ValueError(
                "The argument `parameter_change` must be appropriately positive or negative to span the `parameter_range`"
            )

    if parameter2_value is None and fix_adapt_changes2:
        raise ValueError("If fix_adapt_changes2 is provided, so must parameter2_value")
    if parameter2_value is not None and not fix_adapt_changes2:
        raise ValueError("If parameter2_value is provided, so must fix_adapt_changes2")

    _check_fix_adapt_changes_format(fix_adapt_changes)
    if fix_adapt_changes[0][0] == "pair":
        name1 = "-".join([fix_adapt_changes[0][1][0].replace("/", "-"), fix_adapt_changes[0][1][1]])
    elif fix_adapt_changes[0][0] == "atom":
        name1 = str(fix_adapt_changes[0][1][0])
    else:
        name1 = "kspace"
    if fix_adapt_changes2:
        _check_fix_adapt_changes_format(fix_adapt_changes2)
        if fix_adapt_changes2[0][0] == "pair":
            name2 = "-".join([fix_adapt_changes2[0][1][0].replace("/", "-"), fix_adapt_changes2[0][1][1]])
        elif fix_adapt_changes2[0][0] == "atom":
            name2 = "-".join([str(fix_adapt_changes2[0][1][0]), str(fix_adapt_changes2[0][1][-1] * parameter2_value)])
        else:
            name2 = "kspace"

    file = [
        "\n# Variables and System Conditions\n",
        f"variable TK equal {temperature}\n",
        f"variable freq equal {output_frequency}\n",
        f"variable runtime_equil equal {n_run_equil_steps}\n",
        f"variable runtime_prod equal {n_run_prod_steps}\n",
        "variable pinst equal press\n",
        "variable tinst equal temp\n",
        "variable vinst equal vol\n",
        "variable pe equal pe\n",
        f"variable deltacdm equal {del_parameter} # delta used in central different method for derivative in TI\n",
        f"variable deltacdm2 equal -{del_parameter} # delta used in central different method for derivative in TI\n",
        f"variable nblocks equal {nblocks} \n",
        f"variable lambdas vector [{','.join([str(np.round(x, decimals=prec)) for x in parameter_array])}]\n",
        "\n",
        "thermo ${freq}\n",
        "\n# Set-up Loop\n",
        "variable runid loop 1 ${nblocks} pad\n",
        "    label runloop1\n",
        "\n    # Adjust param for the box and equilibrate\n",
        "    variable param equal v_lambdas[v_runid]\n",
        '    if "${runid} == 1" then &\n',
        '        "jump SELF skipequil"\n',
        "    variable ind equal v_runid-1\n",
        "    variable param0 equal v_lambdas[v_ind]\n",
        "    variable paramramp equal ramp(v_param0,v_param)\n",
    ]
    ind = 13
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            file[ind + 2 * ii : ind + 2 * ii] = [f"variable deltacdmA{ii} equal {args[-1]}*v_deltacdm\n"]
            tmp = str(args[-1])
            if tmp[0] == "-":
                file[ind + 2 * ii + 1 : ind + 2 * ii + 1] = [f"variable deltacdm2A{ii} equal {tmp[1:]}*v_deltacdm\n"]
            else:
                file[ind + 2 * ii + 1 : ind + 2 * ii + 1] = [f"variable deltacdm2A{ii} equal -{tmp}*v_deltacdm\n"]

    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            file.append(f"    variable paramrampA{ii} equal {args[-1]}*ramp(v_param0,v_param)\n")
            file.append(f"    variable paramA{ii} equal {args[-1]}*v_param\n")
    # Add previously changed parameter if it exists
    if fix_adapt_changes2:
        file[ind:ind] = [
            "\n# Set Previous Change\n",
            f"variable param2 equal {parameter2_value}\n",
        ]
        ind += 2
        for ii, (attribute, args) in enumerate(fix_adapt_changes2):
            if attribute == "atom":
                file[ind + ii : ind + ii] = [f"variable param2A{ii} equal {args[-1]}*v_param2\n"]
        ind += ii + 1
        file[ind:ind] = ["fix ADAPT2 all adapt/fep 1 &\n"]
        ind += 1
        for ii, (attribute, args) in enumerate(fix_adapt_changes2):
            if attribute == "atom":
                tmp = " ".join([str(x) for x in args[:-1]])
                file[ind + ii : ind + ii] = [f"    {attribute} {tmp}  v_param2A{ii} &\n"]
            else:
                if attribute == "pair":
                    arg1, arg2 = args[2:]
                    arg1 = str(arg1).replace("*", " ").split(" ")
                    arg2 = str(arg2).replace("*", " ").split(" ")
                    if int(arg2[0]) < int(arg1[0]):
                        args = (args[0], args[1], args[3], args[2])
                tmp = " ".join([str(x) for x in args])
                file[ind + ii : ind + ii] = [f"    {attribute} {tmp}  v_param2 &\n"]
        file[ind + ii] = file[ind + ii].replace("&", "")

    file.append("    fix ADAPT all adapt/fep 1 &\n")
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"        {attribute} {tmp}  v_paramrampA{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1 = str(arg1).replace("*", " ").split(" ")
                arg2 = str(arg2).replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"        {attribute} {tmp}  v_paramramp &\n")
    file[-1] = file[-1].replace("&", "")
    file.extend(
        [
            "    thermo_style custom step v_paramramp "
            + " ".join(
                [f"v_paramrampA{ii}" for ii, (attribute, _) in enumerate(fix_adapt_changes) if attribute == "atom"]
            )
            + " temp press pe evdwl enthalpy\n",
            "    run ${runtime_equil} # Run Ramp\n",
            "    unfix ADAPT\n",
            "    fix ADAPT all adapt/fep ${freq} &\n",
        ]
    )
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"        {attribute} {tmp}  v_paramA{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1 = str(arg1).replace("*", " ").split(" ")
                arg2 = str(arg2).replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"        {attribute} {tmp}  v_param &\n")
    file[-1] = file[-1].replace("&", "")
    file.extend(
        [
            "    thermo_style custom step v_param "
            + " ".join([f"v_paramA{ii}" for ii, (attribute, _) in enumerate(fix_adapt_changes) if attribute == "atom"])
            + " temp press pe evdwl enthalpy\n",
            "    run ${runtime_equil} # Run Equil\n",
            "\n    label skipequil\n\n",
        ]
    )
    if fix_adapt_changes2:
        file.append(f"    write_data files/{name1}_" + "${param}_" + f"{name2}_{parameter2_value}.data\n")
    else:
        file.append(
            f"    write_data files/{name1}_" + "${param}.data\n",
        )
    file.extend(
        [
            "\n    # Initialize computes\n",
            "    thermo_style custom step v_param temp press pe evdwl enthalpy\n",
            "    compute FEPdb all fep ${TK} &\n",
        ]
    )
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"        {attribute} {tmp}  v_deltacdm2A{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1 = str(arg1).replace("*", " ").split(" ")
                arg2 = str(arg2).replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"        {attribute} {tmp}  v_deltacdm2 &\n")
    file.append("        volume yes\n")
    file.append("    compute FEPdf all fep ${TK} &\n")
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"        {attribute} {tmp}  v_deltacdmA{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1 = str(arg1).replace("*", " ").split(" ")
                arg2 = str(arg2).replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"        {attribute} {tmp}  v_deltacdm &\n")
    file.append("        volume yes\n")
    if not fix_adapt_changes2:
        file.extend(
            [
                "    fix FEPout all ave/time ${freq} 1 ${freq} v_param v_deltacdm v_tinst v_pinst v_vinst v_pe &\n",
                f"        c_FEPdb[1] c_FEPdf[1] file files/ti_{name1}_" + "${param}.txt\n\n",
                "    dump TRAJ all custom ${freq} "
                + f"files/dump_{name1}_"
                + "${param}.lammpstrj id mol type element xu yu zu\n",
            ]
        )
    else:
        file.extend(
            [
                "    fix FEPout all ave/time ${freq} 1 ${freq} v_param v_deltacdm v_param2 v_tinst v_pinst v_vinst v_pe &\n",
                f"        c_FEPdb[1] c_FEPdf[1] file files/ti_{name1}_"
                + "${param}_"
                + f"{name2}_{parameter2_value}.txt\n\n",
                "    dump TRAJ all custom ${freq} "
                + f"files/dump_{name1}_"
                + "${param}_"
                + f"{name2}_{parameter2_value}.lammpstrj id mol type element xu yu zu\n",
            ]
        )
    file.extend(
        [
            "\n    run ${runtime_prod}\n\n",
            "    uncompute FEPdb\n",
            "    uncompute FEPdf\n",
            '    if "${runid} != 1" then &\n',
            '        "unfix ADAPT"\n',
            "    unfix FEPout\n",
            "    undump TRAJ\n",
            "\n    next runid\n",
            "    jump SELF runloop1\n",
            "thermo_style custom v_param temp press pe evdwl enthalpy\n",
            "write_data final.data nocoeff\n",
        ]
    )

    if output_file is not None:
        with open(output_file, "w") as f:
            for line in file:
                f.write(line)

    return file


def generate_mbar_input(
    parameter_range,
    parameter_change,
    fix_adapt_changes,
    temperature,
    del_parameter=0.01,
    n_run_equil_steps=1000000,
    n_run_prod_steps=1000000,
    output_frequency=1000,
    output_file=None,
    fix_adapt_changes2=None,
    parameter2_value=None,
    dump=False,
    parameter_array=None,
):
    """Outputs the section of a LAMMPS input file that loops over the values of parameter being changed (e.g., lambda)
    Small perturbations in the potential energy are also output so that the derivative can be calculated for thermodynamic
    integration. Trajectories are produces so that files for MBAR analysis may be generated in post-processing.

    The input data file for this script should be an equilibrated frame in the NPT ensemble. Notice that the input file contains
    the following keywords that you might replace with the values for your simulation using `sed`: TEMP, PRESS

    Parameters
    ----------
    parameter_range : list[float]
        Range of parameter values to be changed where the first value should be the value with which the system has been
        equilibrated.
    parameter_change : float
        The size of the step between parameter values, where a positive value increases and a negative value decreases.
        Take care that number of points needed to traverse the given range should result in an integer, otherwise
        LAMMPS will not end at the desired value.
    fix_adapt_changes : list
        A list of lists containing the following information from `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_:
        attribute (str), args (tuple), e.g., [["pair", ("lj/cut/soft", "lambda", 1, 2)]]. The attributes supported in LAMMPS are:

        - pair; arguments are: pair_style parameter solvent_type(s) solute_type(s)

        - pair_style (str) : String of LAMMPS pair style being changes
        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
        - solvent_type (str) : String defining atom types in the solvent (no spaces), lists are denoted with an asterisk
        - solute_type (str) : String defining atom types in the solute (no spaces), lists are denoted with an asterisk

        - atom; arguments are: parameter atom_type(s). These lines will be scaled by the variation of a "lambda" parameter
            whose range is defined in `parameter_range`, where the starting and end points are multiplied by ``variable_initial``.

        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
        - atom_type (str) : String defining atom types being affected, lists are denoted with an asterisk
        - variable_initial (float) : Specify the initial value of the variable

    temperature : float
        Temperature of the simulation. This will create a variable that is used in the FEP computes.
    del_parameter : float, default=0.1
        Change used to calculate the forward and backward difference used to compute the derivative through a central difference
        approximation. Must be greater then zero.
    n_run_equil_steps : int, default=1000000
        Number of steps in each equilibration run, first with a ramp from the old lambda value to the new value and then for
        equilibration at the new value. The total number of time steps taken per step in lambda (i.e., window) is
        2*n_run_equil_steps + n_run_prod_steps
    n_run_prod_steps : int, default=1000000
        Number of steps in each production run. The total number of time steps taken per step in lambda (i.e., window) is
        2*n_run_equil_steps + n_run_prod_steps
    output_frequency : int, default=1000
        Number of steps between thermo output and dump output.
    output_file : str, default=None
        File name and path for optional output LAMMPS file, otherwise each line is returned as a list.
    fix_adapt_changes2 : list, default=None
        A list of lists containing the following information from `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_:
        attribute (str), args (tuple). This option is useful when a parameter that has been varied and is set to another value
        in this simulation, e.g., lambda when the Coulomb potential is set to zero. Using this feature avoids complications
        with writing the pair potential information in the data file. The attributes supported in LAMMPS are:

        - pair; arguments are: pair_style parameter solvent_type(s) solute_type(s)

        - pair_style (str) : String of LAMMPS pair style being changes
        - parameter (str) : Parameter being varied, see table in `compute fep <https://docs.lammps.org/compute_fep.html>`_
          for the options in your pair-potential
        - solvent_type (str) : String defining atom types in the solvent (no spaces), lists are denoted with an asterisk
        - solute_type (str) : String defining atom types in the solute (no spaces), lists are denoted with an asterisk

        - atom; arguments are: parameter atom_type(s). These lines will be scaled by the variation of a "lambda" parameter
            whose range is defined in `parameter_range`, where the starting and end points are multiplied by ``variable_initial``.

        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options such as "charge"
        - atom_type (str) : String defining atom types being affected, lists are denoted with an asterisk
        - variable_initial (float) : Specify the initial value of the variable

    parameter2_value : float, default=None
        Value to set ``parameter2`` in ``fix_adapt_changes2``. Parameter that has been varied and is set to another value in this simulation, e.g., lambda when the Coulomb potential
        is set to zero. Using this feature avoids complications with writing the pair potential information in the data file.
        See table in `compute fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
    dump : bool, default=False
        If True, trajectories are dumped for each ``parameter`` value.
    parameter_array : list, default=None
        If not ``None``, this argument will overwrite the ``parameter_change`` argument and use the specified steps in
        the parameter. Note that the first value should represent the state that the trajectory is equilibrated at.

    Returns
    -------
    file : list[str]
        List of strings representing lines in a file

    """
    del_parameter = np.abs(del_parameter)

    if parameter_array is None:
        prec = len(repr(parameter_change).split(".")[-1])
        parameter_array = np.concatenate(
            (np.arange(parameter_range[0], parameter_range[1], parameter_change), [parameter_range[1]])
        )
        flag_array = False
    else:
        parameter_array = np.array(parameter_array)
        prec = int(np.max([len(repr(x).split(".")[-1]) for x in parameter_array]))
        flag_array = True

    nblocks = len(parameter_array)
    if nblocks < 2:
        if flag_array:
            raise ValueError(
                "If `parameter_array is not None`, an array of parameters must be provided, instead of {}".format(
                    parameter_array
                )
            )
        else:
            raise ValueError(
                "The argument `parameter_change` must be appropriately positive or negative to span the `parameter_range`"
            )

    if parameter2_value is None and fix_adapt_changes2:
        raise ValueError("If fix_adapt_changes2 is provided, so must parameter2_value")
    if parameter2_value is not None and not fix_adapt_changes2:
        raise ValueError("If parameter2_value is provided, so must fix_adapt_changes2")

    _check_fix_adapt_changes_format(fix_adapt_changes)
    if fix_adapt_changes[0][0] == "pair":
        name1 = "-".join([fix_adapt_changes[0][1][0].replace("/", "-"), fix_adapt_changes[0][1][1]])
    elif fix_adapt_changes[0][0] == "atom":
        name1 = str(fix_adapt_changes[0][1][0])
    else:
        name1 = "kspace"
    if fix_adapt_changes2:
        _check_fix_adapt_changes_format(fix_adapt_changes2)
        if fix_adapt_changes2[0][0] == "pair":
            name2 = "-".join([fix_adapt_changes2[0][1][0].replace("/", "-"), fix_adapt_changes2[0][1][1]])
        elif fix_adapt_changes2[0][0] == "atom":
            name2 = "-".join([str(fix_adapt_changes2[0][1][0]), str(fix_adapt_changes2[0][1][-1] * parameter2_value)])
        else:
            name2 = "kspace"

    file = [
        "\n# Variables and System Conditions\n",
        f"variable TK equal {temperature}\n",
        f"variable freq equal {output_frequency}\n",
        f"variable runtime_equil equal {n_run_equil_steps}\n",
        f"variable runtime_prod equal {n_run_prod_steps}\n",
        "variable pinst equal press\n",
        "variable tinst equal temp\n",
        "variable vinst equal vol\n",
        "variable pe equal pe\n",
        f"variable deltacdm equal {del_parameter} # delta used in central different method for derivative in TI\n",
        f"variable deltacdm2 equal -{del_parameter} # delta used in central different method for derivative in TI\n",
        f"variable nblocks equal {nblocks}\n",
        f"variable lambdas vector [{','.join([str(np.round(x, decimals=prec)) for x in parameter_array])}]\n",
        "\n",
        "thermo ${freq}\n",
        "\n# Set-up Loop\n",
        "variable runid loop 1 ${nblocks} pad\n",
        "    label runloop1\n",
        "\n    # Adjust param for the box and equilibrate\n",
        "    variable param equal v_lambdas[v_runid]\n",
        '    if "${runid} == 1" then &\n',
        '        "jump SELF skipequil"\n',
        "    variable ind equal v_runid-1\n",
        "    variable param0 equal v_lambdas[v_ind]\n",
        "    variable paramramp equal ramp(v_param0,v_param)\n",
    ]
    ind = 13
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            file[ind + 2 * ii : ind + 2 * ii] = [f"variable deltacdmA{ii} equal {args[-1]}*v_deltacdm\n"]
            tmp = str(args[-1])
            if tmp[0] == "-":
                file[ind + 2 * ii + 1 : ind + 2 * ii + 1] = [f"variable deltacdm2A{ii} equal {tmp[1:]}*v_deltacdm\n"]
            else:
                file[ind + 2 * ii + 1 : ind + 2 * ii + 1] = [f"variable deltacdm2A{ii} equal -{tmp}*v_deltacdm\n"]

    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            file.append(f"    variable paramrampA{ii} equal {args[-1]}*ramp(v_param0,v_param)\n")
            file.append(f"    variable paramA{ii} equal {args[-1]}*v_param\n")
    # Add previously changed parameter if it exists
    if fix_adapt_changes2:
        file[ind:ind] = [
            "\n# Set Previous Change\n",
            f"variable param2 equal {parameter2_value}\n",
        ]
        ind += 2
        for ii, (attribute, args) in enumerate(fix_adapt_changes2):
            if attribute == "atom":
                file[ind + ii : ind + ii] = [f"variable param2A{ii} equal {args[-1]}*v_param2\n"]
        ind += ii + 1
        file[ind:ind] = ["fix ADAPT2 all adapt/fep 1 &\n"]
        ind += 1
        for ii, (attribute, args) in enumerate(fix_adapt_changes2):
            if attribute == "atom":
                tmp = " ".join([str(x) for x in args[:-1]])
                file[ind + ii : ind + ii] = [f"    {attribute} {tmp}  v_param2A{ii} &\n"]
            else:
                if attribute == "pair":
                    arg1, arg2 = args[2:]
                    arg1 = str(arg1).replace("*", " ").split(" ")
                    arg2 = str(arg2).replace("*", " ").split(" ")
                    if int(arg2[0]) < int(arg1[0]):
                        args = (args[0], args[1], args[3], args[2])
                tmp = " ".join([str(x) for x in args])
                file[ind + ii : ind + ii] = [f"    {attribute} {tmp}  v_param2 &\n"]
        file[ind + ii] = file[ind + ii].replace("&", "")

    file.append("    fix ADAPT all adapt/fep 1 &\n")
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"        {attribute} {tmp}  v_paramrampA{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1 = str(arg1).replace("*", " ").split(" ")
                arg2 = str(arg2).replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"        {attribute} {tmp}  v_paramramp &\n")
    file[-1] = file[-1].replace("&", "")
    file.extend(
        [
            "    thermo_style custom step v_paramramp "
            + " ".join(
                [f"v_paramrampA{ii}" for ii, (attribute, _) in enumerate(fix_adapt_changes) if attribute == "atom"]
            )
            + " temp press pe evdwl enthalpy\n",
            "    run ${runtime_equil} # Run Ramp\n",
            "    unfix ADAPT\n",
            "    fix ADAPT all adapt/fep ${freq} &\n",
        ]
    )
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"        {attribute} {tmp}  v_paramA{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1 = str(arg1).replace("*", " ").split(" ")
                arg2 = str(arg2).replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"        {attribute} {tmp}  v_param &\n")
    file[-1] = file[-1].replace("&", "")
    file.extend(
        [
            "    thermo_style custom step v_param "
            + " ".join([f"v_paramA{ii}" for ii, (attribute, _) in enumerate(fix_adapt_changes) if attribute == "atom"])
            + " temp press pe evdwl enthalpy\n",
            "    run ${runtime_equil} # Run Equil\n",
            "\n    label skipequil\n\n",
            f"    write_data files/{name1}_" + "${param}.data\n",
            "\n    # Initialize computes\n",
            "    thermo_style custom step v_param temp press pe evdwl enthalpy\n",
        ]
    )
    file.append("    compute FEPdb all fep ${TK} &\n")
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"        {attribute} {tmp}  v_deltacdm2A{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1 = str(arg1).replace("*", " ").split(" ")
                arg2 = str(arg2).replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"        {attribute} {tmp}  v_deltacdm2 &\n")
    file.append("        volume yes\n")
    file.append("    compute FEPdf all fep ${TK} &\n")
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"        {attribute} {tmp}  v_deltacdmA{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1 = str(arg1).replace("*", " ").split(" ")
                arg2 = str(arg2).replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"        {attribute} {tmp}  v_deltacdm &\n")
    file.append("        volume yes\n")
    if not fix_adapt_changes2:
        file.extend(
            [
                "    fix FEPout all ave/time ${freq} 1 ${freq} v_param v_deltacdm v_tinst v_pinst v_vinst v_pe &\n",
                f"        c_FEPdb[1] c_FEPdf[1] file files/ti_{name1}_" + "${param}.txt\n\n",
            ]
        )
    else:
        file.extend(
            [
                "    fix FEPout all ave/time ${freq} 1 ${freq} v_param v_deltacdm v_param2 v_tinst v_pinst v_vinst v_pe &\n",
                f"        c_FEPdb[1] c_FEPdf[1] file files/ti_{name1}_"
                + "${param}_"
                + f"{name2}_{parameter2_value}.txt\n\n",
            ]
        )

    # Write out the free energy difference between the current state and the other states
    for i in range(nblocks):
        file.append("    variable delta{0:0d} equal ".format(i) + f"v_lambdas[{i+1}]-v_param\n")
        for ii, (attribute, args) in enumerate(fix_adapt_changes):
            if attribute == "atom":
                file.append(
                    "    variable deltaA{0:0d}".format(i) + f"{ii} equal {args[-1]}*" + "v_delta{0:0d}\n".format(i)
                )
        file.append("    compute FEP{0:03d} all fep ".format(i) + "${TK} &\n")
        for ii, (attribute, args) in enumerate(fix_adapt_changes):
            if attribute == "atom":
                tmp = " ".join([str(x) for x in args[:-1]])
                file.append(f"        {attribute} {tmp}  " + "v_deltaA{0:0d}".format(i) + f"{ii} &\n")
            else:
                if attribute == "pair":
                    arg1, arg2 = args[2:]
                    arg1 = str(arg1).replace("*", " ").split(" ")
                    arg2 = str(arg2).replace("*", " ").split(" ")
                    if int(arg2[0]) < int(arg1[0]):
                        args = (args[0], args[1], args[3], args[2])
                tmp = " ".join([str(x) for x in args])
                file.append(f"        {attribute} {tmp}  " + "v_delta{0:0d} &\n".format(i))
        file.extend(
            [
                "        volume yes\n",
                "    variable param{0:03d} equal v_param+v_delta{0:0d}\n".format(i),
            ]
        )
        if fix_adapt_changes2:
            file.extend(
                [
                    "    fix FEPout{0:03d} all".format(i)
                    + " ave/time ${freq} 1 ${freq} "
                    + "v_param v_param{0:03d} v_param2 v_pe &\n".format(i),
                    "        c_FEP{0:03d}[1] c_FEP{0:03d}[2] c_FEP{0:03d}[3]".format(i)
                    + f" file files/mbar_{name1}"
                    + "_${param}_${param"
                    + str("{0:03d}".format(i))
                    + "}_"
                    + "{}.txt\n\n".format(name2),
                ]
            )
        else:
            file.extend(
                [
                    "    fix FEPout{0:03d} all".format(i)
                    + " ave/time ${freq} 1 ${freq} "
                    + "v_param v_param{0:03d} v_pe &\n".format(i),
                    "        c_FEP{0:03d}[1] c_FEP{0:03d}[2] c_FEP{0:03d}[3]".format(i)
                    + f" file files/mbar_{name1}"
                    + "_${param}_${param"
                    + str("{0:03d}".format(i))
                    + "}.txt\n\n",
                ]
            )
    if not fix_adapt_changes2:
        file.append(
            "    dump TRAJ all custom ${freq} "
            + f"files/dump_{name1}_"
            + "${param}.lammpstrj id mol type element xu yu zu\n"
        )
    else:
        file.append(
            "    dump TRAJ all custom ${freq} "
            + f"files/dump_{name1}_"
            + "${param}_"
            + f"{name2}_{parameter2_value}.lammpstrj id mol type element xu yu zu\n"
        )
    file.extend(
        [
            "\n    run ${runtime_prod}\n\n",
            "    uncompute FEPdb\n",
            "    uncompute FEPdf\n",
            '    if "${runid} != 1" then &\n',
            '        "unfix ADAPT"\n',
        ]
    )
    for i in range(nblocks):
        file.extend(
            [
                "    uncompute FEP{0:03d}\n".format(i),
                "    unfix FEPout{0:03d}\n".format(i),
            ]
        )
    file.extend(
        [
            "    unfix FEPout\n",
            "    undump TRAJ\n",
            "\n    next runid\n",
            "    jump SELF runloop1\n",
            "thermo_style custom v_param temp press pe evdwl enthalpy\n",
            "write_data final.data nocoeff\n",
        ]
    )
    if not dump:  # Comment out trajectroy lines
        file = [x if "TRAJ" not in x else x[:2] + "#" + x[2:] for x in file]

    if output_file is not None:
        with open(output_file, "w") as f:
            for line in file:
                f.write(line)

    return file


def generate_rerun_mbar(
    parameter_value,
    parameter_range,
    parameter_change,
    fix_adapt_changes,
    temperature,
    output_frequency=1000,
    output_file=None,
    fix_adapt_changes2=None,
    parameter2_value=None,
    parameter_array=None,
):
    """Outputs the section of a LAMMPS input file that reruns trajectories for different lambda values and calculates
    the potential energy for all other lambda values with this set of configurations.

    Parameters
    ----------
    parameter_value : float
        Value of parameter being varied (e.g., lambda)
    parameter_range : list[float]
        Range of parameter values to be changed where the first value should be the value with which the system has been
        equilibrated.
    parameter_change : float
        The size of the step between parameter values, where a positive value increases and a negative value decreases.
        Take care that number of points needed to traverse the given range should result in an integer, otherwise
        LAMMPS will not end at the desired value.
    fix_adapt_changes : list
        A list of lists containing the following information from `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_:
        attribute (str), args (tuple), e.g., [["pair", ("lj/cut/soft", "lambda", 1, 2)]]. The attributes supported in LAMMPS are:

        - pair; arguments are: pair_style parameter solvent_type(s) solute_type(s)

        - pair_style (str) : String of LAMMPS pair style being changes
        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
        - solvent_type (str) : String defining atom types in the solvent (no spaces), lists are denoted with an asterisk
        - solute_type (str) : String defining atom types in the solute (no spaces), lists are denoted with an asterisk

        - atom; arguments are: parameter atom_type(s). These lines will be scaled by the variation of a "lambda" parameter
            whose range is defined in `parameter_range`, where the starting and end points are multiplied by ``variable_initial``.

        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
        - atom_type (str) : String defining atom types being affected, lists are denoted with an asterisk
        - variable_initial (float) : Specify the initial value of the variable

    temperature : float
        Temperature of the simulation. This will create a variable that is used in the FEP computes.
    output_frequency : int, default=1000
        Number of steps between thermo output and dump output.
    output_file : str, default=None
        File name and path for optional output file
    fix_adapt_changes2 : list, default=None
        A list of lists containing the following information from `fix adapt/fep <https://docs.lammps.org/fix_adapt_fep.html>`_:
        attribute (str), args (tuple). This option is useful when a parameter that has been varied and is set to another value
        in this simulation, e.g., lambda when the Coulomb potential is set to zero. Using this feature avoids complications
        with writing the pair potential information in the data file. The attributes supported in LAMMPS are:

        - pair; arguments are: pair_style parameter solvent_type(s) solute_type(s)

        - pair_style (str) : String of LAMMPS pair style being changes
        - parameter (str) : Parameter being varied, see table in `compute fep <https://docs.lammps.org/compute_fep.html>`_
          for the options in your pair-potential
        - solvent_type (str) : String defining atom types in the solvent (no spaces), lists are denoted with an asterisk
        - solute_type (str) : String defining atom types in the solute (no spaces), lists are denoted with an asterisk

        - atom; arguments are: parameter atom_type(s). These lines will be scaled by the variation of a "lambda" parameter
            whose range is defined in `parameter_range`, where the starting and end points are multiplied by ``variable_initial``.

        - parameter (str) : Parameter being varied, see table in `compute
          fep <https://docs.lammps.org/compute_fep.html>`_ for the options such as "charge"
        - atom_type (str) : String defining atom types being affected, lists are denoted with an asterisk
        - variable_initial (float) : Specify the initial value of the variable

    parameter2_value : float, default=None
        Value to set ``parameter2`` in ``fix_adapt_changes2``. Parameter that has been varied and is set to another value in this simulation, e.g., lambda when the Coulomb potential
        is set to zero. Using this feature avoids complications with writing the pair potential information in the data file.
        See table in `compute fep <https://docs.lammps.org/compute_fep.html>`_ for the options in your pair-potential
    parameter_array : list, default=None
        If not ``None``, this argument will overwrite the ``parameter_change`` argument and use the specified steps in
        the parameter. Note that the first value should represent the state that the trajectory is equilibrated at.

    Returns
    -------
    file : list[str]
        List of strings representing lines in a file

    """
    if parameter_array is None:
        prec = len(repr(parameter_change).split(".")[-1])
        parameter_array = np.concatenate(
            (np.arange(parameter_range[0], parameter_range[1], parameter_change), [parameter_range[1]])
        )
        flag_array = False
    else:
        parameter_array = np.array(parameter_array)
        prec = int(np.max([len(repr(x).split(".")[-1]) for x in parameter_array]))
        flag_array = True

    nblocks = len(parameter_array)
    if nblocks < 2:
        if flag_array:
            raise ValueError(
                "If `parameter_array is not None`, an array of parameters must be provided, instead of {}".format(
                    parameter_array
                )
            )
        else:
            raise ValueError(
                "The argument `parameter_change` must be appropriately positive or negative to span the `parameter_range`"
            )

    if parameter2_value is None and fix_adapt_changes2:
        raise ValueError("If fix_adapt_changes2 is provided, so must parameter2_value")
    if parameter2_value is not None and not fix_adapt_changes2:
        raise ValueError("If parameter2_value is provided, so must fix_adapt_changes2")

    _check_fix_adapt_changes_format(fix_adapt_changes)
    if fix_adapt_changes[0][0] == "pair":
        name1 = "-".join([fix_adapt_changes[0][1][0].replace("/", "-"), fix_adapt_changes[0][1][1]])
    elif fix_adapt_changes[0][0] == "atom":
        name1 = str(fix_adapt_changes[0][1][0])
    else:
        name1 = "kspace"
    if fix_adapt_changes2:
        _check_fix_adapt_changes_format(fix_adapt_changes2)
        if fix_adapt_changes2[0][0] == "pair":
            name2 = "-".join([fix_adapt_changes2[0][1][0].replace("/", "-"), fix_adapt_changes2[0][1][1]])
        elif fix_adapt_changes2[0][0] == "atom":
            name2 = "-".join([str(fix_adapt_changes2[0][1][0]), str(fix_adapt_changes2[0][1][-1] * parameter2_value)])
        else:
            name2 = "kspace"

    file = [
        "\n# Variables and System Conditions\n",
        f"variable TK equal {temperature}\n",
        f"variable freq equal {output_frequency}\n",
        "variable pinst equal press\n",
        "variable tinst equal temp\n",
        "variable vinst equal vol\n",
        "variable pe equal pe\n",
        f"variable param equal {parameter_value}\n",
        f"variable lambdas vector [{','.join([str(np.round(x, decimals=prec)) for x in parameter_array])}]\n",
        "\nthermo ${freq}\n",
        "\n# Initialize computes\n",
    ]
    ind = 10
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            file.append(f"    variable paramA{ii} equal {args[-1]}*v_param\n")
    # Add previously changed parameter if it exists
    if fix_adapt_changes2:
        file[ind:ind] = [
            "\n# Set Previous Change\n",
            f"variable param2 equal {parameter2_value}\n",
        ]
        ind += 2
        for ii, (attribute, args) in enumerate(fix_adapt_changes2):
            if attribute == "atom":
                file[ind + ii : ind + ii] = [f"variable param2A{ii} equal {args[-1]}*v_param2\n"]
        ind += ii + 1
        file[ind:ind] = ["fix ADAPT2 all adapt/fep 1 &\n"]
        ind += 1
        for ii, (attribute, args) in enumerate(fix_adapt_changes2):
            if attribute == "atom":
                tmp = " ".join([str(x) for x in args[:-1]])
                file[ind + ii : ind + ii] = [f"    {attribute} {tmp}  v_param2A{ii} &\n"]
            else:
                if attribute == "pair":
                    arg1, arg2 = args[2:]
                    arg1 = str(arg1).replace("*", " ").split(" ")
                    arg2 = str(arg2).replace("*", " ").split(" ")
                    if int(arg2[0]) < int(arg1[0]):
                        args = (args[0], args[1], args[3], args[2])
                tmp = " ".join([str(x) for x in args])
                file[ind + ii : ind + ii] = [f"    {attribute} {tmp}  v_param2 &\n"]
        file[ind + ii] = file[ind + ii].replace("&", "")

    file.append("fix ADAPT all adapt/fep ${freq} &\n")
    for ii, (attribute, args) in enumerate(fix_adapt_changes):
        if attribute == "atom":
            tmp = " ".join([str(x) for x in args[:-1]])
            file.append(f"    {attribute} {tmp}  v_paramA{ii} &\n")
        else:
            if attribute == "pair":
                arg1, arg2 = args[2:]
                arg1 = str(arg1).replace("*", " ").split(" ")
                arg2 = str(arg2).replace("*", " ").split(" ")
                if int(arg2[0]) < int(arg1[0]):
                    args = (args[0], args[1], args[3], args[2])
            tmp = " ".join([str(x) for x in args])
            file.append(f"    {attribute} {tmp}  v_param &\n")
    file[-1] = file[-1].replace("&", "\n")

    for i in range(nblocks):
        delta = np.round(parameter_array[i] - parameter_value, decimals=prec)
        file.append("variable delta{0:0d}".format(i) + " equal {}\n".format(delta))
        for ii, (attribute, args) in enumerate(fix_adapt_changes):
            if attribute == "atom":
                file.append("variable deltaA{0:0d}".format(i) + f"{ii} equal {args[-1]}*" + "v_delta{0:0d}\n".format(i))
        file.append("compute FEP{0:03d} all fep ".format(i) + "${TK} &\n")
        for ii, (attribute, args) in enumerate(fix_adapt_changes):
            if attribute == "atom":
                tmp = " ".join([str(x) for x in args[:-1]])
                file.append(f"    {attribute} {tmp}  " + "v_deltaA{0:0d}".format(i) + f"{ii} &\n")
            else:
                if attribute == "pair":
                    arg1, arg2 = args[2:]
                    arg1 = str(arg1).replace("*", " ").split(" ")
                    arg2 = str(arg2).replace("*", " ").split(" ")
                    if int(arg2[0]) < int(arg1[0]):
                        args = (args[0], args[1], args[3], args[2])
                tmp = " ".join([str(x) for x in args])
                file.append(f"    {attribute} {tmp}  " + "v_delta{0:0d} &\n".format(i))
        file.extend(
            [
                "    volume yes\n",
                "variable param{0:03d} equal v_param+v_delta{0:0d}\n".format(i),
            ]
        )
        if fix_adapt_changes2:
            file.extend(
                [
                    "fix FEPout{0:03d} all".format(i)
                    + " ave/time ${freq} 1 ${freq} "
                    + "v_param v_param{0:03d} v_param2 v_pe &\n".format(i),
                    "    c_FEP{0:03d}[1] c_FEP{0:03d}[2] c_FEP{0:03d}[3]".format(i)
                    + f" file files/mbar_{name1}"
                    + "_${param}_${param"
                    + str("{0:03d}".format(i))
                    + "}_"
                    + "{}.txt\n\n".format(name2),
                ]
            )
        else:
            file.extend(
                [
                    "    fix FEPout{0:03d} all".format(i)
                    + " ave/time ${freq} 1 ${freq} "
                    + "v_param v_param{0:03d} v_pe &\n".format(i),
                    "        c_FEP{0:03d}[1] c_FEP{0:03d}[2] c_FEP{0:03d}[3]".format(i)
                    + f" file files/mbar_{name1}"
                    + "_${param}_${param"
                    + str("{0:03d}".format(i))
                    + "}.txt\n\n",
                ]
            )
    if not fix_adapt_changes2:
        file.append(f"\nrerun files/dump_{name1}_" + "${param}.lammpstrj every ${freq} dump x y z\n\n")
    else:
        file.append(
            f"\nrerun files/dump_{name1}_"
            + "${param}_"
            + f"{name2}_{parameter2_value}.lammpstrj "
            + "every ${freq} dump x y z\n\n"
        )

    if output_file is not None:
        with open(output_file, "w") as f:
            for line in file:
                f.write(line)

    return file
