### HISTORY OF CHANGES

##### November 19, 2025 (Release of Version 1.0)

Stable version 1.0 checked, validated and released.

* Python 2 support has been dropped. Only Python 3 compatibility will be developed and maintained from now on.
* The library has been reshaped to be compliant with the modern [PyPA specifications](https://packaging.python.org/en/latest/specifications/).
* [Hatch](https://hatch.pypa.io/latest/) was chosen as the tool to build and publish the package. See the *pyproject.toml* file. 
* Bug fixes to adapt to the various changes in Python and NumPy since last release.


#### January 14, 2022 (version 0.2.0)
There is no more minimum bound imposed on the estimated weights by default. If you want to estimate positive weights, use min_C=0 in the fit method! 


