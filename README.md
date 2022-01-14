# pyMOU
#### Light package for simulation and connectivity estimation using the multivariate Ornstein-Uhlenbeck process (MOU)


This Python library relies on the MOU process to simulate network activity and estimate connectivity from observed activity (Gilson et al. *PLoS Comput Biol* 2016; Gilson et al *Net Neurosci* 2020).


## NEW IN THIS VERSION

There is no more minimum bound imposed on the estimated weights by default. If you want to estimate positive weights, use min_C=0 in the fit method! 


## Quick install using pip 

The installer pip comes with Anaconda and most Python distribution. Just run:

    $ pip install git+https://github.com/mb-BCA/pyMOU.git@master

## How to use?

Check the Python notebooks in `examples`.
