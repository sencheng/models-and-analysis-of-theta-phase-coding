Created by Eloy Parra Barrero; eloyparra.b@gmail.com; 2021-05-21

Code for the analyses included in:
### Neuronal sequences during theta rely on behavior-dependent spatial maps (Parra-Barrero, E., Diba, K., Cheng, S., 2021)

The files in the main folder define the classes used for the analyses included in the paper (and a bit more).
The files in ./analyze run the analyses. ./analyze also contains the following folders:

- parameters - contains parameter files (paper.json defines the parameters used in the paper)
- fields - contains the info defining the manually screened place fields
- sessions - contains files with general info / parameters specific to each experimental session

#### Usage:
1. Download the datasets hc-3 and hc-11 from crncs.org
1. Change the paths defined in ./analyze/config.py
1. Run individual analyses (e.g., ./analyze/run_firing_fields.py) or run a batch of analyses.

For running a batch of analyses, first choose what to analyze in ./analyze/batch_config.py,
then run ./analyze/batch_analyses.py and plot the results with ./analyze/batch_plots.py.
batch_plots.py produces all of the summary figures of the paper.

Some of the objects created for the analyses require a lot of computation, so they are saved as pickles and loaded again
when required, unless the code or the parameters of the object or one of its dependencies has changed, in which case
a new object is created.

To discard the manually screened fields and screen them again, delete the corresponding file(s) in ./analyze/fields and
run any code that will initialize a FiringFields instance (e.g., ./analyze/batch_analyses.py). However, this will not
work if an old pickle of FiringFields will be loaded, so delete the pickles or introduce some dummy change in the code
or in the parameters to trigger a new instance of FiringFields to be initialized.
