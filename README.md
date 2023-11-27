[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vpicouet/fireball2-etc/main?labpath=notebooks%2FETC.ipynb)

Click on the video to access to the ETC
[![Alt Text](https://github.com/vpicouet/pyds9plugin-doc/blob/master/docs/fig/ETC.gif)](https://www.youtube.com/watch?v=XcDm2JQDMLY)
You can add your instrument characteristics [here](https://docs.google.com/spreadsheets/d/1Ox0uxEm2TfgzYA6ivkTpU4xrmN5vO5kmnUPdCSt73uU/edit?usp=sharing)

[![Alt Text](https://github.com/vpicouet/pyds9plugin-doc/blob/master/docs/fig/Instruments.jpg)](https://docs.google.com/spreadsheets/d/1Ox0uxEm2TfgzYA6ivkTpU4xrmN5vO5kmnUPdCSt73uU/edit?usp=sharing)




# <center>ETC - SNR calculator </center>

## Description
This straightforward ETC provides an estimated signal-to-noise ratio for an extended source on a resolution element. Its key attributes lie in its versatility, user-friendly interface, and diverse plotting options. The ETC is designed to accommodate any spectrograph, and individuals can effortlessly contribute new instruments or configurations [here](https://docs.google.com/spreadsheets/d/1Ox0uxEm2TfgzYA6ivkTpU4xrmN5vO5kmnUPdCSt73uU/edit?pli=1#gid=2066284077) for direct utilization within the tool.

The ETC GUI serves as a valuable resource, offering a quick overview of not only the essential instrument parameters and constraints but also presenting insights into tradeoffs, potential optimizations, and mitigation strategies. Currently, there are two primary visualization options: the SNR plot and the image simulation, both capable of accommodating all instruments stored in the [spreadsheet](https://docs.google.com/spreadsheets/d/1Ox0uxEm2TfgzYA6ivkTpU4xrmN5vO5kmnUPdCSt73uU/edit?pli=1#gid=2066284077). These visualizations dynamically adjust as parameters are modified using the intuitive widgets.


## Variables

The different variables used which can all be used in the x axis to analyze the impact on SNR are:
- **Source:** flux, sky, source extension, source's line width
- **Observing strategy:** Exposure time, total acquisition time, atmospheric transmission, observed wavelength, distance to source/line center
- **Instrument design:** collecting area, plate scale, instrument throughput, spatial resolution (at the mask and at the detector)
- **Spectrograph design:** spectral resolution, slit width, dispersion
- **Detector parameters:** quantum efficiency, dark current, read noise, readout time, pixel size, image loss due to cosmic ray
- **emCCD additional parameters:** EM gain, CIC, thresholding, smearing exponential length and temperature if you check it (based on a first rough evolution of smearing and dark current with temperature, therefore changing the temperature will change smearing and dark accordingly.)
- **Image simulator related parameters:** Full well of the detector, conversion gain and throughput FWHM to add the λ dependancy.



## Outputs

### SNR visualization
The SNR plot is crafted to offer a straightforward depiction of noise budgets concerning various variables that can be fine-tuned or mitigated to enhance instrument sensitivity. In the top panel, the noise from distinct sources (Signal, Dark, Sky, CIC, RN) is presented in electrons per pixel. The middle panel provides the average electron-per-pixel value for each component (pre-stacking). The last plot outlines the relative fractions of all noise sources per resolution element per N frames over the total acquisition tim and the resulting SNR.

### Image simulator

The image simulator utilizes the various parameters and a specific source (galaxy/stellar spectra) to simulate the following:
- **single & stacked image:** Presented in the upper left and right sections, these images are 100 × 500 pixels (resulting in distinct physical FOVs for different instruments). The spectral direction is horizontal, while the spatial one is vertical. The slit size is incorporated, along with contributions from different noise sources.
- **Histogram:** Lower left plot displays the histogram for both individual and stacked images.
- **Profiles:** Lower right plot offers profiles in both spatial and spectral directions for the single (large & transparent) and stacked images.

The code assumes a standard (λ-dependent) atmospheric transmission for ground instruments. Users are encouraged to upload their instrument throughput/QE λ-dependency on the GitHub repository (under notebook/interpolate) using the format "Instrument_name.csv" (λ in nanometers on the first column and with no column name). If no table is added, the code will default to using the Throughput_FWHM value in the spreadsheet.

# <center>Contributions calculations in Electrons per pixel </center>

## Sky and signal
The sky and signal contributions are first converted from $ergs/cm^2/s/asec^2/Å$ to photons/cm $^2$/s/sr/Å (continuum unit): $CU =   \frac{Flux}{\frac{h c}{ \lambda} \times \frac{\pi}{ 180 \times 3600}^2 }$

Note the wavelength dependancy in the formula. 
We decide deliberately to use flux per Angstrom with a gaussian profile so that the user can simulate both a continuum or an unresolved line. 
Users can also directly upload spectra in  $ergs/cm^2/s/asec^2/Å$ in the GitHub repository (under notebooks/Spectra, λ in nanometers on the first column and with no column name).

Then, both contributions are converted similarly into electrons per pixels:





$$Sky_{e-/pix/exp} = Sky_{CU} \times Slitwidth_{str}  \times Dispersion_{Å/pix}  \times Texp_s \times Atm_{\%}\times  Area_{\%} \times Throughput_{\%}  \times QE_{\%}  $$

$$Signal_{e-/pix/exp} = Signal_{CU} \times Slitwidth_{str}  \times Dispersion_{Å/pix} \times Texp_s \times Atm_\%\times  Area_\% \times Throughput_{\%}  \times QE_{\%}  $$

<!-- If the instrument is an imager (no slit and no dispersion), we replace the factor $ Slitwidth_{str}  \times Dispersion_{Å/pix}$ by $FOV_{str} \times Bandwidth_{Å} $. -->

## Other contributions

Other contributions (dark current, read-noise, CIC, straylight) are easier to account for.
Dark and straylight are used with the same unit: $e-/pix/hour$. Therefore:

$Dark_{e-/pix/exp} = Dark_{e-/pix/hour} \times \frac{ Texp_{s} }{3600}$

CIC (Clock induced charges) which are charges induced in electron amplified CCD, are already given in e-/pix/exp.
Read noise is usually also given in e-/pix/exp.

## <center>Conversion to noise </center>


Each contribution is then converted to noise by taking the square root of the contribution and accounting for the effective number of frames and the element resolution size: 

$N_{Contribution} = \sqrt{Contribution [\times ENF] \times N_{images} \times Size_{resolution element} }$

The number of effective images is:

$$N_{images} = \frac{Ttot_{s}}{Texp_{s} + Tread_{s}} \times (1-CRloss_{\%}) $$

In the case of electron amplified CCDs, some considerations must be taken into account:
- the read noise must be divided by the amplification gain: $RN_{e-/pix/exp} = \frac{ RN_{e-/pix/exp} }{EMGain_{e-/e-}}$
- an excess noise factor of $\sqrt{2}$  must be used to account for the stochastic amplification (if no thresholding method is applied)




<!-- Other considerations taken into account:
- cosmic ray loss directly impacting the number of images 
- Taking into account line width
- Taking into account cut my the slit
- 

When doing some photon-counting emCCD thresholding, some new considerations must be added:
- Only part of the charges are accounted for
- When thresholding is applied, the smearing can be changed as it will impact the position of the threshold that optimizes the SNR. For this optimal threshold, the fraction of signal and readnoise kept (above the threshold) is given.
-  -->











![alternative text](description/Chart.jpg)

