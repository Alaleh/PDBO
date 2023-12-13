This code is tested using `python 3.8`

Installing the dependencies:

`pip install -r requirements.txt`

Example of a command to run the code:

`python scripts/run.py --algo PDBO --problem zdt2 --n-var 4 --n-obj 2 --batch-size 2 4 8 16 --n-seed 25`

The problem name, batch size, the number of variable, and the number of the objectives can be changed. You can enter multiple batch sizes to evaluate in parallel.
All results are saved in the '\result' folder.

The problem name is selected from the following options: zdt1, zdt2, zdt3, dtlz1,  dtlz3,  dtlz5, gtd

The HV results per iteration are computed during the optimization and are included in the results files.

In order to calculate the PFD, after obtaining the results use the following command:

`python scripts/calculateParetoFrontDiversity.py`

