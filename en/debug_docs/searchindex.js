Search.setIndex({"docnames": ["README", "_autosummary/generate_alchemical_lammps_inputs", "_autosummary/generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs", "_autosummary/generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_input_linear_approximation", "_autosummary/generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_mbar_input", "_autosummary/generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_rerun_mbar", "_autosummary/generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_traj_input", "_templates/README", "api", "authors", "changelog", "contributing", "index", "license", "readme"], "filenames": ["README.md", "_autosummary/generate_alchemical_lammps_inputs.rst", "_autosummary/generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.rst", "_autosummary/generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_input_linear_approximation.rst", "_autosummary/generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_mbar_input.rst", "_autosummary/generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_rerun_mbar.rst", "_autosummary/generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_traj_input.rst", "_templates/README.md", "api.rst", "authors.rst", "changelog.rst", "contributing.rst", "index.rst", "license.rst", "readme.rst"], "titles": ["Compiling Generate Alchemical LAMMPS Inputs\u2019s Documentation", "generate_alchemical_lammps_inputs", "generate_alchemical_lammps_inputs", "generate_input_linear_approximation", "generate_mbar_input", "generate_rerun_mbar", "generate_traj_input", "Templates Doc Directory", "API Documentation", "Authors", "Changelog for Generate Alchemical LAMMPS Inputs", "How to Contribute", "Generate Alchemical LAMMPS Inputs\u2019s Documentation", "License", "Getting Started"], "terms": {"The": [0, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14], "doc": [0, 13], "thi": [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14], "project": [0, 10, 12, 14], "ar": [0, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14], "built": 0, "sphinx": [0, 7], "reinstal": [0, 12, 14], "packag": [0, 10, 11, 12, 14], "depend": [0, 3], "from": [0, 2, 3, 4, 5, 6, 11, 12, 14], "root": 0, "directori": [0, 12, 14], "pip": [0, 12, 14], "instal": 0, "onc": 0, "you": [0, 3, 4, 6, 11, 12, 13, 14], "can": [0, 2, 3, 4, 6], "us": [0, 2, 3, 4, 5, 6, 11, 12, 14], "makefil": 0, "static": 0, "html": [0, 7, 10], "page": [0, 7, 11, 12, 14], "make": [0, 12, 13, 14], "_build": 0, "view": 0, "open": [0, 11], "index": [0, 12], "which": [0, 3, 4, 5, 6], "mai": [0, 4, 6, 12, 13, 14], "itself": 0, "insid": 0, "call": 0, "what": 0, "version": [0, 10], "i": [0, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14], "descript": [1, 2], "function": [1, 2, 12, 14], "gener": [1, 2, 4, 6, 11], "lammp": [1, 2, 3, 4, 5, 6], "input": [1, 2, 3, 4, 5, 6], "alchem": [1, 2], "calcul": [1, 2, 4, 5, 6, 12, 14], "For": 2, "clariti": 2, "we": [2, 11], "would": 2, "like": [2, 7], "distinguish": 2, "differ": [2, 4, 5, 6], "between": [2, 3, 4, 5, 6], "lambda": [2, 3, 4, 5, 6], "lambda_2": 2, "refer": 2, "scale": [2, 3, 4, 5, 6], "factor": 2, "potenti": [2, 3, 4, 5, 6], "energi": [2, 3, 4, 5, 6, 12, 14], "equilibr": [2, 3, 4, 5, 6], "system": [2, 3, 4, 5, 6, 12, 14], "so": [2, 4, 6, 7], "when": [2, 3, 4, 5, 6, 11], "valu": [2, 3, 4, 5, 6], "chang": [2, 3, 4, 5, 6, 10, 12, 13, 14], "undergo": 2, "anoth": [2, 3, 4, 5, 6], "step": [2, 3, 4, 5, 6, 12, 14], "On": [2, 11], "other": [2, 5, 10, 11], "hand": 2, "despit": 2, "those": 2, "configur": [2, 5], "being": [2, 3, 4, 5, 6], "two": 2, "instanc": 2, "first": [2, 3, 4, 5, 6], "thermodynam": [2, 3, 4, 6, 12, 14], "integr": [2, 3, 4, 6, 11, 12, 14], "ti": [2, 3], "veri": 2, "close": 2, "deriv": [2, 4, 6, 13], "free": [2, 3, 12, 13, 14], "respect": 2, "need": [2, 3, 4, 5, 6], "becaus": 2, "doe": [2, 12, 13, 14], "comput": [2, 3, 4, 5, 6], "explicit": 2, "although": 2, "one": [2, 10, 11], "should": [2, 3, 4, 5, 6, 13], "check": [2, 11], "whether": 2, "thei": [2, 7], "an": [2, 3, 4, 5, 6], "express": [2, 13], "cannot": 2, "soft": [2, 3, 4, 5, 6], "lennard": 2, "jone": 2, "lj": [2, 3, 4, 5, 6], "file": [2, 3, 4, 5, 6, 9, 10, 12, 13, 14], "cumbersom": 2, "have": [2, 11], "been": [2, 3, 4, 5, 6], "includ": [2, 12, 13, 14], "appropri": [2, 13], "section": [2, 3, 4, 5, 6], "If": [2, 3, 4, 5, 6, 11, 12, 14], "linear": 2, "approxim": [2, 4, 6], "made": 2, "u_": 2, "post": [2, 4, 6], "process": [2, 4, 6, 12, 14], "recommend": [2, 12, 14], "generate_input_linear_approxim": 2, "run": [2, 3, 4, 6, 11, 12, 14], "loop": [2, 4, 6], "over": [2, 4, 6], "all": [2, 3, 5, 9, 10, 11, 12, 13, 14], "save": 2, "frame": [2, 3, 4, 6], "space": [2, 3, 4, 5, 6], "independ": 2, "sampl": 2, "output": [2, 3, 4, 5, 6], "small": [2, 4, 6], "perturb": [2, 4, 6], "achiev": 2, "generate_traj_input": 2, "after": [2, 7, 11], "simul": [2, 3, 4, 5, 6, 12, 14], "mbar": [2, 3, 4, 6, 12, 14], "rerun": [2, 5], "featur": [2, 3, 4, 5, 6, 11], "break": 2, "up": 2, "allow": [2, 12, 14], "add": [2, 3, 11], "addit": [2, 12], "point": [2, 3, 4, 5, 6], "analysi": [2, 4, 6], "without": [2, 12, 13, 14], "repeat": 2, "initi": [2, 3, 4, 5, 6, 12, 14], "generate_rerun_mbar": 2, "notic": [2, 3, 4, 6, 13], "do": 2, "contain": [2, 3, 4, 5, 6, 7], "header": 2, "inform": [2, 3, 4, 5, 6], "expect": 2, "specif": 2, "left": 2, "user": 2, "note": [2, 3, 4, 5, 6, 11, 13], "fix": [2, 3, 4, 5, 6, 10], "adapt": [2, 3, 4, 5, 6], "fep": [2, 3, 4, 5, 6], "generate_alchemical_lammps_input": [3, 4, 5, 6, 11, 12, 13, 14], "parameter_rang": [3, 4, 5, 6], "parameter_chang": [3, 4, 5, 6], "fix_adapt_chang": [3, 4, 5, 6], "temperatur": [3, 4, 5, 6], "types_solut": 3, "types_solv": 3, "n_run_equil_step": [3, 4, 6], "1000000": [3, 4, 6], "n_run_prod_step": [3, 4, 6], "output_frequ": [3, 4, 5, 6], "1000": [3, 4, 5, 6], "output_fil": [3, 4, 5, 6], "none": [3, 4, 5, 6], "fix_adapt_changes2": [3, 4, 5, 6], "parameter2_valu": [3, 4, 5, 6], "is_charg": 3, "true": [3, 4], "parameter_arrai": [3, 4, 5, 6], "sourc": [3, 4, 5, 6, 13], "separ": 3, "coulomb": [3, 4, 5, 6], "nonbond": 3, "bond": 3, "angl": 3, "torsion": 3, "contribut": [3, 9, 12, 14], "solut": [3, 4, 5, 6], "solvent": [3, 4, 5, 6], "As": 3, "long": 3, "paramet": [3, 4, 5, 6], "linearli": 3, "each": [3, 4, 6, 10, 11], "multi": [3, 12, 14], "state": [3, 4, 5, 6, 12, 13, 14], "bennett": [3, 12, 14], "accept": [3, 12, 14], "ratio": [3, 12, 14], "data": [3, 4, 5, 6, 12, 13, 14], "script": [3, 4, 6, 12, 14], "npt": [3, 4, 6], "ensembl": [3, 4, 6], "follow": [3, 4, 5, 6, 10, 12, 14], "keyword": [3, 4, 6], "might": [3, 4, 6], "replac": [3, 4, 6], "your": [3, 4, 5, 6, 11, 12, 14], "sed": [3, 4, 6], "temp": [3, 4, 6], "press": [3, 4, 6], "list": [3, 4, 5, 6], "float": [3, 4, 5, 6], "rang": [3, 4, 5, 6], "where": [3, 4, 5, 6, 13], "ha": [3, 4, 5, 6], "size": [3, 4, 5, 6], "posit": [3, 4, 5, 6], "increas": [3, 4, 5, 6], "neg": [3, 4, 5, 6], "decreas": [3, 4, 5, 6], "take": [3, 4, 5, 6], "care": [3, 4, 5, 6], "number": [3, 4, 5, 6], "travers": [3, 4, 5, 6], "given": [3, 4, 5, 6], "result": [3, 4, 5, 6, 11, 13], "integ": [3, 4, 5, 6], "otherwis": [3, 4, 5, 6], "end": [3, 4, 5, 6], "desir": [3, 4, 5, 6], "A": [3, 4, 5, 6, 9, 11, 12, 13, 14], "attribut": [3, 4, 5, 6], "str": [3, 4, 5, 6], "arg": [3, 4, 5, 6], "tupl": [3, 4, 5, 6], "e": [3, 4, 5, 6, 12, 14], "g": [3, 4, 5, 6], "pair": [3, 4, 5, 6], "cut": [3, 4, 5, 6], "1": [3, 4, 5, 6, 12, 14], "2": [3, 4, 5, 6, 12, 14], "support": [3, 4, 5, 6], "argument": [3, 4, 5, 6], "pair_styl": [3, 4, 5, 6], "solvent_typ": [3, 4, 5, 6], "": [3, 4, 5, 6, 7, 11], "solute_typ": [3, 4, 5, 6], "string": [3, 4, 5, 6], "style": [3, 4, 5, 6], "vari": [3, 4, 5, 6], "see": [3, 4, 5, 6], "tabl": [3, 4, 5, 6], "option": [3, 4, 5, 6, 12, 14], "defin": [3, 4, 5, 6], "atom": [3, 4, 5, 6], "type": [3, 4, 5, 6], "denot": [3, 4, 5, 6], "asterisk": [3, 4, 5, 6], "atom_typ": [3, 4, 5, 6], "These": [3, 4, 5, 6, 12, 14], "line": [3, 4, 5, 6, 7, 12, 14], "variat": [3, 4, 5, 6], "whose": [3, 4, 5, 6], "start": [3, 4, 5, 6], "multipli": [3, 4, 5, 6], "variable_initi": [3, 4, 5, 6], "affect": [3, 4, 5, 6], "specifi": [3, 4, 5, 6], "variabl": [3, 4, 5, 6], "creat": [3, 4, 5, 6, 11, 12, 13, 14], "colon": 3, "int": [3, 4, 5, 6], "default": [3, 4, 5, 6], "ramp": [3, 4, 6], "old": [3, 4, 6], "new": [3, 4, 6, 11, 12, 14], "total": [3, 4, 6], "time": [3, 4, 6], "taken": [3, 4, 6], "per": [3, 4, 6], "window": [3, 4, 6, 12, 14], "product": [3, 4, 6], "thermo": [3, 4, 5, 6], "dump": [3, 4, 5, 6], "name": [3, 4, 5, 6, 7, 9, 11], "path": [3, 4, 5, 6, 7], "set": [3, 4, 5, 6, 7], "zero": [3, 4, 5, 6], "avoid": [3, 4, 5, 6], "complic": [3, 4, 5, 6], "write": [3, 4, 5, 6], "charg": [3, 4, 5, 6], "parameter2": [3, 4, 5, 6], "bool": [3, 4], "kspace": 3, "compon": 3, "record": 3, "overwrit": [3, 4, 5, 6, 7], "repres": [3, 4, 5, 6, 13], "trajectori": [3, 4, 5, 6], "return": [3, 4, 5, 6, 11], "del_paramet": [4, 6], "0": [4, 6], "01": [4, 6, 9], "fals": 4, "also": [4, 6], "produc": [4, 6, 12, 14], "forward": [4, 6], "backward": [4, 6], "through": [4, 6], "central": [4, 6], "must": [4, 11], "greater": 4, "parameter_valu": 5, "ani": [7, 13], "here": [7, 11], "rel": 7, "conf": 7, "py": 7, "copi": [7, 13], "builtin": 7, "folder": 7, "html_static_path": 7, "_templat": [7, 13], "extens": 7, "stock": 7, "layout": 7, "below": 9, "repositori": [9, 11, 12, 14], "histori": 9, "changelog": [9, 12], "show": 9, "individu": 9, "code": [9, 11], "email": 9, "github": [9, 11, 12, 13, 14], "id": 9, "orcid": 9, "nist": 9, "ou": 9, "divis": [9, 12, 14], "group": [9, 12, 14], "jennif": [9, 12, 14], "clark": [9, 12, 14], "gov": [9, 12, 14], "jaclark5": 9, "0000": 9, "0003": 9, "4897": 9, "5651": 9, "materi": [9, 12, 14], "measur": [9, 12, 14], "laboratori": [9, 12, 14], "642": 9, "scienc": [9, 12, 14], "engin": [9, 12, 14], "polym": [9, 12, 14], "complex": [9, 12, 14], "fluid": [9, 12, 14], "notabl": 10, "document": [10, 11], "under": 10, "subhead": 10, "ad": [10, 11, 12, 14], "deprec": 10, "remov": 10, "secur": 10, "perform": 10, "adher": 10, "semant": 10, "http": [10, 12, 14], "semver": 10, "org": [10, 12, 14], "spec": 10, "v2": 10, "creation": 10, "welcom": 11, "extern": 11, "contributor": 11, "merg": 11, "sure": 11, "account": 11, "fork": 11, "local": [11, 12, 13, 14], "machin": 11, "clone": [11, 12, 14], "branch": [11, 12, 14], "idea": 11, "relat": 11, "go": 11, "readi": 11, "examin": 11, "comment": 11, "navig": 11, "pull": 11, "request": 11, "pr": 11, "launch": 11, "subsequ": 11, "commit": [11, 12, 14], "automat": 11, "valid": 11, "mergabl": 11, "compil": 11, "test": [11, 12, 14], "suit": 11, "complianc": [11, 13], "visibl": 11, "re": 11, "provid": [11, 13], "case": 11, "pytest": 11, "consid": 11, "box": 11, "let": 11, "dev": 11, "know": 11, "complet": [11, 12, 14], "until": 11, "continu": 11, "checkmark": 11, "multipl": 11, "core": 11, "develop": [11, 12, 13, 14], "give": 11, "approv": 11, "review": 11, "best": [11, 12, 14], "practic": 11, "guid": 11, "softwar": [11, 12, 14], "think": 11, "exampl": 11, "api": 12, "how": 12, "resourc": 12, "author": 12, "certain": [12, 14], "commerci": [12, 14], "equip": [12, 13, 14], "instrument": [12, 14], "identifi": [12, 14], "paper": [12, 14], "foster": [12, 14], "understand": [12, 14], "Such": [12, 14], "identif": [12, 14], "impli": [12, 13, 14], "endors": [12, 14], "nation": [12, 13, 14], "institut": [12, 13, 14], "standard": [12, 13, 14], "technologi": [12, 13, 14], "nor": [12, 13, 14], "necessarili": [12, 14], "avail": [12, 14], "purpos": [12, 13, 14], "wa": [12, 14], "public": [12, 13, 14], "hydrat": [12, 14], "solvat": [12, 14], "water": [12, 14], "solubl": [12, 14], "bar": [12, 14], "command": [12, 14], "python": [12, 14], "m": [12, 14], "d": [12, 14], "onlin": [12, 14], "3": [12, 13, 14], "10": [12, 14], "maco": [12, 14], "linux": [12, 14], "No": [12, 14], "librari": [12, 14], "requir": [12, 14], "befor": [12, 14], "download": [12, 14], "master": [12, 14], "our": [12, 14], "zip": [12, 14], "work": [12, 13, 14], "git": [12, 14], "com": [12, 14], "usnistgov": [12, 14], "conda": [12, 14], "want": [12, 14], "environ": [12, 14], "env": [12, 14], "f": [12, 14], "yaml": [12, 14], "flag": [12, 14], "4": [12, 14], "pre": [12, 14], "supersed": [12, 14], "most": [12, 14], "updat": [12, 14], "languag": [12, 14], "access": [12, 14], "research": [12, 14], "copyright": [12, 13, 14], "fair": [12, 14], "statement": [12, 14], "srd": [12, 14], "phd": [12, 14], "debra": [12, 14], "j": [12, 14], "audu": [12, 14], "jack": [12, 14], "dougla": [12, 14], "analyt": [12, 14], "2024": [12, 14], "doi": [12, 14], "18434": [12, 14], "mds2": [12, 14], "3641": [12, 14], "modul": [12, 13], "search": 12, "servic": 13, "distribut": 13, "medium": 13, "keep": 13, "intact": 13, "entir": 13, "improv": 13, "modifi": 13, "portion": 13, "modif": 13, "carri": 13, "date": 13, "natur": 13, "pleas": 13, "explicitli": 13, "expressli": 13, "AS": 13, "NO": 13, "warranti": 13, "OF": 13, "kind": 13, "IN": 13, "fact": 13, "OR": 13, "aris": 13, "BY": 13, "oper": 13, "law": 13, "limit": 13, "THE": 13, "merchant": 13, "fit": 13, "FOR": 13, "particular": 13, "non": 13, "infring": 13, "AND": 13, "accuraci": 13, "neither": 13, "warrant": 13, "THAT": 13, "WILL": 13, "BE": 13, "uninterrupt": 13, "error": 13, "defect": 13, "correct": 13, "NOT": 13, "represent": 13, "regard": 13, "thereof": 13, "BUT": 13, "TO": 13, "reliabl": 13, "sole": 13, "respons": 13, "determin": 13, "assum": 13, "risk": 13, "associ": 13, "its": 13, "cost": 13, "program": 13, "applic": 13, "damag": 13, "loss": 13, "unavail": 13, "interrupt": 13, "intend": 13, "situat": 13, "failur": 13, "could": 13, "caus": 13, "injuri": 13, "properti": 13, "employe": 13, "subject": 13, "protect": 13, "within": 13, "unit": 13, "autosummari": 13, "rst": 13, "mit": 13, "class": 13, "bsd": 13, "claus": 13}, "objects": {"": [[1, 0, 0, "-", "generate_alchemical_lammps_inputs"]], "generate_alchemical_lammps_inputs": [[2, 0, 0, "-", "generate_alchemical_lammps_inputs"]], "generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs": [[3, 1, 1, "", "generate_input_linear_approximation"], [4, 1, 1, "", "generate_mbar_input"], [5, 1, 1, "", "generate_rerun_mbar"], [6, 1, 1, "", "generate_traj_input"]]}, "objtypes": {"0": "py:module", "1": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"]}, "titleterms": {"compil": 0, "gener": [0, 10, 12, 14], "alchem": [0, 10, 12, 14], "lammp": [0, 10, 12, 14], "input": [0, 10, 12, 14], "": [0, 12], "document": [0, 8, 12, 14], "generate_alchemical_lammps_input": [1, 2], "generate_input_linear_approxim": 3, "generate_mbar_input": 4, "generate_rerun_mbar": 5, "generate_traj_input": 6, "templat": 7, "doc": 7, "directori": 7, "exampl": 7, "file": 7, "add": 7, "thi": 7, "api": 8, "author": 9, "chronolog": 9, "list": 9, "2024": [9, 10], "changelog": 10, "0": 10, "how": 11, "contribut": 11, "get": [11, 12, 14], "start": [11, 12, 14], "make": 11, "chang": 11, "addit": 11, "resourc": 11, "content": 12, "nist": [12, 13, 14], "disclaim": [12, 14], "depend": [12, 14], "instal": [12, 14], "licens": [12, 13, 14], "contact": [12, 14], "affili": [12, 14], "citat": [12, 14], "indic": 12, "tabl": 12, "softwar": 13, "statement": 13, "code": 13, "us": 13, "acknowledg": 13}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx.ext.intersphinx": 1, "nbsphinx": 4, "sphinx": 58}, "alltitles": {"Compiling Generate Alchemical LAMMPS Inputs\u2019s Documentation": [[0, "compiling-generate-alchemical-lammps-inputs-s-documentation"]], "generate_alchemical_lammps_inputs": [[1, "generate-alchemical-lammps-inputs"], [2, "generate-alchemical-lammps-inputs"]], "generate_input_linear_approximation": [[3, "generate-input-linear-approximation"]], "generate_mbar_input": [[4, "generate-mbar-input"]], "generate_rerun_mbar": [[5, "generate-rerun-mbar"]], "generate_traj_input": [[6, "generate-traj-input"]], "Templates Doc Directory": [[7, "templates-doc-directory"]], "Examples of file to add to this directory": [[7, "examples-of-file-to-add-to-this-directory"]], "API Documentation": [[8, "api-documentation"]], "Authors": [[9, "authors"]], "Chronological List of Authors": [[9, "chronological-list-of-authors"]], "2024": [[9, "id1"]], "Changelog for Generate Alchemical LAMMPS Inputs": [[10, "changelog-for-generate-alchemical-lammps-inputs"]], "0.0.0 (2024)": [[10, "id1"]], "How to Contribute": [[11, "how-to-contribute"]], "Getting Started": [[11, "getting-started"], [12, "getting-started"], [14, "getting-started"]], "Making Changes": [[11, "making-changes"]], "Additional Resources": [[11, "additional-resources"]], "Generate Alchemical LAMMPS Inputs\u2019s Documentation": [[12, "generate-alchemical-lammps-inputs-s-documentation"]], "Contents:": [[12, null]], "NIST Disclaimer": [[12, "nist-disclaimer"], [14, "nist-disclaimer"]], "Generate Alchemical LAMMPS Inputs": [[12, "generate-alchemical-lammps-inputs"], [14, "generate-alchemical-lammps-inputs"]], "Documentation": [[12, "documentation"], [14, "documentation"]], "Dependencies": [[12, "dependencies"], [14, "dependencies"]], "Installation": [[12, "installation"], [14, "installation"]], "LICENSE": [[12, "license"], [14, "license"]], "Contact": [[12, "contact"], [14, "contact"]], "Affiliation": [[12, "affiliation"], [14, "affiliation"]], "Citation": [[12, "citation"], [14, "citation"]], "Indices and Tables": [[12, "indices-and-tables"]], "License": [[13, "license"]], "NIST Software Licensing Statement": [[13, "nist-software-licensing-statement"]], "Code-Use Acknowledgements": [[13, "code-use-acknowledgements"]]}, "indexentries": {"generate_alchemical_lammps_inputs": [[1, "module-generate_alchemical_lammps_inputs"]], "module": [[1, "module-generate_alchemical_lammps_inputs"], [2, "module-generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs"]], "generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs": [[2, "module-generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs"]], "generate_input_linear_approximation() (in module generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs)": [[3, "generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_input_linear_approximation"]], "generate_mbar_input() (in module generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs)": [[4, "generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_mbar_input"]], "generate_rerun_mbar() (in module generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs)": [[5, "generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_rerun_mbar"]], "generate_traj_input() (in module generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs)": [[6, "generate_alchemical_lammps_inputs.generate_alchemical_lammps_inputs.generate_traj_input"]]}})