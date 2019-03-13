# Meta-Heuristic-Feature-Selection

The only algorithm provided at the moment is a genetic algorithm that using sampling from 
a linearly decreasing pmf (over the population ranked by fitness) as the selection operator.

Everything custom to your problem such as model training, creating training/test splits, etc...
should be done within user-defined fitness function.  Due to the nature of parallelization in
python, a script file (with __main__ and all) will need to be written and called from command prompt
for this program to work.
An example is provided in test.py
