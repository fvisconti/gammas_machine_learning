# Some machine learning in gamma ray astronomy

Gamma ray astronomy is the astronomical observation of gamma rays, the most energetic form of electromagnetic radiation, with photon energies above 100 keV. Radiation below 100 keV is classified as X-rays and is the subject of X-ray astronomy.

When gamma rays reach the earth’s atmosphere they interact with it, producing cascades of subatomic particles. These cascades are also known as air or particle showers. Nothing can travel faster than the speed of light in a vacuum, but light travels 0.03 percent slower in air. Thus, **these ultra-high energy particles can travel faster than light in air**, creating a blue flash of “Cherenkov light”.

Cherenkov light is not only produced by gammas, but by hadrons also; this kind of light is called *background*.
A good background suppression (identify all events detected by the camera generated by hadrons and suppress them) is crucial for progress in this field: here's where **machine learning** comes in.

More on [CTA website](https://www.cta-observatory.org).

## Code
The code is organized as follows:
- `astriml_classes.py` has all the classes needed for the analysis;
- `astriml.py` is a sample script to perform training and save the trained model;
- `astrimlreco.py` is a sample script to perform reconstruction.

Logic in `astriml_classes.py` needs be improved: there's too much branching, due to a previous version written in `C++` with the `Shark` libraries.

## I/O
Input/Output files are in `*.fits` format.

## Configuration
All is highly configurable via [IRAF](https://heasarc.gsfc.nasa.gov/lheasoft/headas/pil/node12.html) parameter files system.

For example, in `astriml.par` one can set:
- Random Forest classificator/regressor settings;
- training parameters names, which requires now the **event** keywords, for example: 
```
gh_sk_par4, s, h, "log10(events['SIZE']/(events['WIDTH']*events['LENGTH']))",,,"4th parameter for g/h RF"
```
- column position index (1 based) for binning parameters (and `log` values for `SIZE`):
```
size_pos_sk, i, h, 1,,, "column position of size feature"
n_resz_sk,  i, h, 100,1,,"Number of log bins for resizing"
lo_resz_sk, r, h, 1,0,,"log10 lowest value of binning for resizing (phe)"
hi_resz_sk, r, h, 6,0,,"log10 highest value of binning for resizing (phe)"
 ```
*Zenith* and *Azimuth* also want column positions, `npar+1` and `npar+2` respectively, with `npar` = total number of parameters used for g-h separation
 
Similar considerations hold for `astrimlreco.par`.

## Trained models
Models are saved via `sklearn.externals.joblib` utility, this means it will be correctly read only with the same `scikit-learn`'s version used for its creation: the best practice for this is as usual to create a `conda` environment to perform the analysis.

## Credits
The code uses `pyAstriPar.py` written by @elehcim to handle IRAF *parfiles*.
