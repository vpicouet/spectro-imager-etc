#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:32:46 2024

@author: Vincent
"""

import os
from astropy.io import fits
import matplotlib.pyplot as plt

# Define the folder containing the FITS images
folder_path = '/Users/Vincent/DS9QuickLookPlugIn/subsets/CIT_NUVU_m100_darks_2022/test/'

# Get a list of all FITS files in the folder
fits_files = [f for f in os.listdir(folder_path) if f.endswith('.fits')]
fits_files.sort()
# Create a subplot with two columns
fig, axs = plt.subplots(2, 1, figsize=(12,10),sharex=True)

# Loop through each FITS file
for i, fits_file in enumerate(fits_files):
    # Open the FITS file
    with fits.open(os.path.join(folder_path, fits_file)) as hdul:
        # Read the data from the first HDU (assuming image data is in the first HDU)
        data = hdul[0].data
        print(hdul[0].header["EXPTIME"])
        # Plot the histogram of the first 1000 columns in log scale
        
        axs[0].hist(data[:, :1000].flatten(), bins=500, range=(1000,5000),log=True,histtype='step',label=hdul[0].header["EMGAIN"])
        # axs[0].set_title(fits_file + ' - First 1000 Columns')
        
        # Plot the histogram of the next 1000 columns in log scale
        axs[1].hist(data[:, 1000:2000].flatten(), bins=500, range=(1000,5000),log=True,histtype='step',label=hdul[0].header["EMGAIN"])
        # axs[1].set_title(fits_file + ' - Next 1000 Columns')
axs[0].legend()
axs[1].legend()
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

#%%

CIC = 0.01
dark = 0.1/3600
seconds = np.linspace(0,3600,100000)
charges = CIC+dark*seconds
plt.plot(seconds,charges)
plt.plot(seconds,CIC*np.ones(len(seconds)))
plt.xscale("log")
# plt.yscale("log")
plt.ylabel("Charges")
plt.xlabel("time")

plt.grid()
plt.show()


#%%

CIC = 0.01
dark = 0.1/3600
seconds = np.linspace(0,600,100000)
noise = np.sqrt(CIC**2+(dark*seconds)**2)
plt.plot(seconds,noise)
plt.plot(seconds,CIC*np.ones(len(seconds)))
# plt.xscale("log")
# plt.yscale("log")

plt.ylabel("noise")
plt.xlabel("time")
plt.grid()
plt.show()


#%%

from scipy.integrate import quad
import numpy as np

# Define the exponentiated function
def exp_func(x, a):
    return np.exp(-x/a)

# Integrate the function from 0 to infinity for different values of a
a_values = [0.5, 1.5,2,5]  # Example values of a, one less than 1 and other greater than 1

for a in a_values:
    result, error = quad(exp_func, 0, np.inf, args=(a,))
    print(f"The integral of exp(-x/{a}) from 0 to infinity is approximately {result} (Error estimate: {error})")
#%%

from astropy.table import Table
import warnings
warnings.filterwarnings("ignore")
# import matplotlib
# matplotlib.use('Agg')

from ipywidgets import Button, Layout, jslink, IntText, IntSlider, interactive, interact, HBox, Layout, VBox
from astropy.modeling.functional_models import Gaussian2D, Gaussian1D
import os
from IPython.display import display, clear_output
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import inspect
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
import inspect
from scipy.sparse import dia_matrix
from scipy.interpolate import interpn
from scipy.special import erf
from astropy.modeling.functional_models import Gaussian2D
import pandas as pd
import functools
from scipy import special

np.seterr(invalid='ignore')
 

sheet_id = "1Ox0uxEm2TfgzYA6ivkTpU4xrmN5vO5kmnUPdCSt73uU"
sheet_name = "instruments.csv"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
try:
    instruments = Table.from_pandas(pd.read_csv(url))
except Exception:
    instruments = Table.read("Instruments.csv")
instruments = instruments[instruments.colnames]
instruments_dict ={ name:{key:float(val) for key, val in zip(instruments["Charact."][:],instruments[name][:])} for name in instruments.colnames[3:]}




def float_to_latex(mumber):
    try:
        return "$\,"+ ("%.1E"%(mumber)).replace("E"," \,10^{")+"}$"
    except TypeError as e:
        print(e,mumber)
    # return ("%.1E"%(mumber)).replace("E"," 10$^{")+"}$"

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def convert_LU2ergs(LU,wave_nm):
    wave =wave_nm * 1e-7 #/ (1+redshift)
    Energy = 6.62e-27 * 3e10 / wave
    angle =  np.pi / (180 * 3600)
    flux_ergs = LU * Energy * angle * angle
    return flux_ergs

def convert_ergs2LU(flux_ergs,wave_nm):
    wave =wave_nm * 1e-7 #/ (1+redshift)
    Energy = 6.62e-27 * 3e10 / wave
    angle =    np.pi / (180 * 3600) 
    LU = flux_ergs/ (Energy  * angle * angle)
    # flux_ergs = LU * Energy * angle * angle
    return LU

def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    # names, varargs, keywords, defaults = inspect.getargspec(func)
    names, varargs, keywords, defaults,_,_,_ = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


# Initialization of the thresholding functions. So that files are not read several times
n=10
type_="" #"new_" #""
#new is for when we don't use fraction and use RN (false I think), "" is with fraction true positives and RN/gain, seems better 
path=""
# path = "/Users/Vincent/Github/fireball2-etc/notebooks/"
table_threshold = fits.open(path+"interpolate/%sthreshold_%s.fits"%(type_,n))[0].data
table_snr = fits.open(path+"interpolate/%ssnr_max_%s.fits"%(type_,n))[0].data
table_fraction_rn = fits.open(path+"interpolate/%sfraction_rn_%s.fits"%(type_,n))[0].data
table_fraction_flux = fits.open(path+"interpolate/%sfraction_flux_%s.fits"%(type_,n))[0].data




def variable_smearing_kernels(image, smearing=1.5, SmearExpDecrement=50000):
    """Creates variable smearing kernels for inversion
    """
    import numpy as np
    
    smearing_length = smearing * np.exp(-image / SmearExpDecrement)
    smearing_kernels = np.exp(-np.arange(6)[:, np.newaxis, np.newaxis] / smearing_length)
    smearing_kernels /= smearing_kernels.sum(axis=0)
    return smearing_kernels   

#temperature=-100,
class Observation:
    @initializer
    # def __init__(self, instrument="FIREBall-2 2023", Atmosphere=0.5, Throughput=0.13*0.9, exposure_time=50, counting_mode=False, Signal=1e-16, EM_gain=1400, RN=109, CIC_charge=0.005, Dard_current=0.08, Sky=10000, readout_time=1.5, extra_background = 0,acquisition_time = 2,smearing=0,i=25,plot_=False,temperature=-100,n=n,PSF_RMS_mask=5, PSF_RMS_det=8, QE = 0.45,cosmic_ray_loss_per_sec=0.005,PSF_source=16,lambda_stack=1,Slitwidth=5,Bandwidth=200,Collecting_area=1,Δx=0,Δλ=0,pixel_scale=np.nan, Spectral_resolution=np.nan, dispersion=np.nan,Line_width=np.nan,wavelength=np.nan, pixel_size=np.nan,len_xaxis=50):#,photon_kept=0.7#, flight_background_damping = 0.9
    def __init__(self, instrument="FIREBall-2 2023", Atmosphere=0.5, Throughput=0.13, exposure_time=50, counting_mode=False, Signal=1e-17, EM_gain=1500, RN=40, CIC_charge=0.005, Dard_current=1, Sky=2e-18, readout_time=5, extra_background = 0.5,acquisition_time = 2,smearing=0.50,i=25,plot_=False,n=n,PSF_RMS_mask=2.5, PSF_RMS_det=3, QE = 0.4,cosmic_ray_loss_per_sec=0.005,PSF_source=16,lambda_stack=0.21,Slitwidth=6,Bandwidth=100,Collecting_area=0.707,Δx=0,Δλ=0,pixel_scale=1.1, Spectral_resolution=1300, dispersion=0.21,Line_width=15,wavelength=200, pixel_size=13,len_xaxis=50,Slitlength=10):#,photon_kept=0.7#, flight_background_damping = 0.9
        """
        ETC calculator: computes the noise budget at the detector level based on instrument/detector parameters
        This is currently optimized for slit spectrographs and EMCCD but could be pretty easily generalized to other instrument type if needed
        """
        self.initilize()
    
    def initilize(self):
        self.precise = True
        self.Signal = Gaussian2D(amplitude=self.Signal,x_mean=0,y_mean=0,x_stddev=self.PSF_source,y_stddev=4,theta=0)(self.Δx,self.Δλ)
        # print("\nAtmosphere",self.Atmosphere, "\nThroughput=",self.Throughput,"\nSky=",Sky, "\nacquisition_time=",acquisition_time,"\ncounting_mode=",counting_mode,"\nSignal=",Signal,"\nEM_gain=",EM_gain,"RN=",RN,"CIC_charge=",CIC_charge,"Dard_current=",Dard_current,"\nreadout_time=",readout_time,"\n_smearing=",smearing,"\nextra_background=",extra_background,"\ntemperature=",temperature,"\nPSF_RMS_mask=",PSF_RMS_mask,"\nPSF_RMS_det=",PSF_RMS_det,"\nQE=",QE,"\ncosmic_ray_loss_per_sec=",self.cosmic_ray_loss_per_sec,"\nlambda_stack",self.lambda_stack,"\nSlitwidth",self.Slitwidth, "\nBandwidth",self.Bandwidth,"\nPSF_source",self.PSF_source,"\nCollecting_area",self.Collecting_area)
        # print("\Collecting_area",self.Collecting_area, "\nΔx=",self.Δx,"\nΔλ=",Δλ, "\napixel_scale=",pixel_scale,"\nSpectral_resolution=",Spectral_resolution,"\ndispersion=",dispersion,"\nLine_width=",Line_width,"wavelength=",wavelength,"pixel_size=",pixel_size)
        # Simple hack to me able to use UV magnitudes (not used for the ETC)
        if np.max([self.Signal])>1:
            self.Signal = 10**(-(self.Signal-20.08)/2.5)*2.06*1E-16
        #TODO be sure we account for potential 2.35 ratio here
        #convolve input flux by instrument PSF
        if self.precise:
            self.Signal *= (erf(self.PSF_source / (2 * np.sqrt(2) * self.PSF_RMS_det)) )
            #convolve input flux by spectral resolution
            # self.spectro_resolution_A = self.wavelength * self.spectral
            self.Signal *= (erf(self.Line_width / (2 * np.sqrt(2) * 10*self.wavelength/self.Spectral_resolution)) )


        if ~np.isnan(self.Slitwidth).all() & self.precise:
            # assess flux fraction going through slit
            self.flux_fraction_slit = (1+erf(self.Slitwidth/(2*np.sqrt(2)*self.PSF_RMS_mask)))-1
        else:
            self.flux_fraction_slit = 1
        # if self.smearing>0:
        self.Signal *= 1 - np.exp(-1/(self.smearing+1e-15)) - np.exp(-2/(self.smearing+1e-15))  - np.exp(-3/(self.smearing+1e-15))
        self.resolution_element= self.PSF_RMS_det * 2.35 /self.pixel_scale  # in pix (before it was in arcseconds)
        self.PSF_lambda_pix = self.wavelength / self.Spectral_resolution / self.dispersion

        red, blue, violet, yellow, green, pink, grey  = '#E24A33','#348ABD','#988ED5','#FBC15E','#8EBA42','#FFB5B8','#777777'
        # self.colors= ['#E24A33','#348ABD','#988ED5','#FBC15E','#FFB5B8','#8EBA42','#777777']
        # self.colors= ['#E24A33','#348ABD','#988ED5','#FBC15E','#8EBA42','#FFB5B8','#777777']
        self.colors= [red, violet, yellow  ,blue, green, pink, grey ]
        # self.Sky_CU =  convert_ergs2LU(self.Sky_,self.wavelength)
        # self.Sky_ = self.Sky_CU*self.lu2ergs# ergs/cm2/s/arcsec^2 

        self.ENF = 1 if self.counting_mode else 2 # Excess Noise Factor 
        self.CIC_noise = np.sqrt(self.CIC_charge * self.ENF) 
        self.Dark_current_f = self.Dard_current * self.exposure_time / 3600 # e/pix/frame
        self.Dark_current_noise =  np.sqrt(self.Dark_current_f * self.ENF)
        
        # For now we put the regular QE without taking into account the photon kept fracton, because then infinite loop. 
        # Two methods to compute it: interpolate_optimal_threshold & compute_optimal_threshold
        self.pixel_size_arcsec = self.pixel_scale
        # self.pixel_scale  = (self.pixel_scale*np.pi/180/3600) #go from arcsec/pix to str/pix 
        self.arcsec2str = (np.pi/180/3600)**2
        self.Sky_CU = convert_ergs2LU(self.Sky, self.wavelength) 
        # self.Sky_ = convert_LU2ergs(self.Sky_CU, self.wavelength) 
        # self.Collecting_area *= 100 * 100#m2 to cm2
        # TODO use astropy.unit
        if self.counting_mode:
            self.factor_CU2el =  self.QE * self.Throughput * self.Atmosphere  *    (self.Collecting_area * 100 * 100)  * self.Slitwidth * self.arcsec2str  * self.dispersion
            self.sky = self.Sky_CU*self.factor_CU2el*self.exposure_time  # el/pix/frame
            self.Sky_noise_pre_thresholding = np.sqrt(self.sky * self.ENF) 
            self.signal_pre_thresholding = self.Signal*self.factor_CU2el*self.exposure_time  # el/pix/frame
            self.n_threshold, self.Photon_fraction_kept, self.RN_fraction_kept, self.gain_thresholding = self.interpolate_optimal_threshold(plot_=self.plot_, i=self.i)#,flux=self.signal_pre_thresholding)
            # self.n_threshold, self.Photon_fraction_kept, self.RN_fraction_kept, self.gain_thresholding = self.compute_optimal_threshold(plot_=plot_, i=i,flux=self.signal_pre_thresholding)
        else:
            self.n_threshold, self.Photon_fraction_kept, self.RN_fraction_kept, self.gain_thresholding = np.zeros(self.len_xaxis),np.ones(self.len_xaxis),np.ones(self.len_xaxis), np.zeros(self.len_xaxis)
        # The faction of detector lost by cosmic ray masking (taking into account ~5-10 impact per seconds and around 2000 pixels loss per impact (0.01%))
        self.cosmic_ray_loss = np.minimum(self.cosmic_ray_loss_per_sec*(self.exposure_time+self.readout_time/2),1)
        self.QE_efficiency = self.Photon_fraction_kept * self.QE
        # TODO verify that indeed it should not depend on self.pixel_scale**2 
        # Compute ratio to convert CU to el/pix 
        if np.isnan(self.Slitwidth).all():
            # If instrument is not a spectro?
            self.factor_CU2el = self.QE_efficiency * self.Throughput * self.Atmosphere  *    (self.Collecting_area * 100 * 100)   * self.Bandwidth  * self.arcsec2str *self.pixel_scale**2 #but here it's total number of electrons we don't know if it is per A or not and so if we need to devide by dispersion: 1LU/A = .. /A. OK So we need to know if sky is LU or LU/A            
        else:
            self.factor_CU2el = self.QE_efficiency * self.Throughput * self.Atmosphere  *    (self.Collecting_area * 100 * 100)  * self.Slitwidth * self.arcsec2str  * self.dispersion *self.pixel_scale**2
        

        self.sky = self.Sky_CU*self.factor_CU2el*self.exposure_time  # el/pix/frame
        self.Sky_noise = np.sqrt(self.sky * self.ENF) 
            
        # TODO in counting mode, Photon_fraction_kept should also be used for CIC
        self.RN_final = self.RN  * self.RN_fraction_kept / self.EM_gain 
        self.Additional_background = self.extra_background/3600 * self.exposure_time# e/pix/exp
        self.Additional_background_noise = np.sqrt(self.Additional_background * self.ENF)
        
        # number of images taken during one field acquisition (~2h)
        self.N_images = self.acquisition_time*3600/(self.exposure_time + self.readout_time)
        self.N_images_true = self.N_images * (1-self.cosmic_ray_loss)

        self.Signal_LU = convert_ergs2LU(self.Signal,self.wavelength)
        # if 1==0: # if line is totally resolved (for cosmic web for instance)
        #     self.Signal_el =  self.Signal_LU*self.factor_CU2el*self.exposure_time * self.flux_fraction_slit  / self.spectral_resolution_pixel # el/pix/frame#     Signal * (sky / Sky_)  #el/pix
        # else: # if line is unresolved for QSO for instance
        self.Signal_el =  self.Signal_LU * self.factor_CU2el * self.exposure_time * self.flux_fraction_slit   # el/pix/frame#     Signal * (sky / Sky_)  #el/pix
        # print(self.flux_fraction_slit)

        self.signal_noise = np.sqrt(self.Signal_el * self.ENF)     #el / resol/ N frame

        self.N_resol_element_A = self.lambda_stack / self.dispersion# / (1/self.dispersion)#/ (10*self.wavelength/self.Spectral_resolution) # should work even when no spectral resolution
        self.factor = np.sqrt(self.N_images_true) * self.resolution_element * np.sqrt(self.N_resol_element_A)
        self.Signal_resolution = self.Signal_el * self.factor**2# el/N exposure/resol
        self.signal_noise_nframe = self.signal_noise * self.factor
        self.Total_noise_final = self.factor*np.sqrt(self.signal_noise**2 + self.Dark_current_noise**2  + self.Additional_background_noise**2 + self.Sky_noise**2 + self.CIC_noise**2 + self.RN_final**2   ) #e/  pix/frame
        self.SNR = self.Signal_resolution / self.Total_noise_final
        
        if type(self.Total_noise_final + self.Signal_resolution) == np.float64:
            n=0
        else:
            n =len(self.Total_noise_final + self.Signal_resolution) 
        if n>1:
            for name in ["signal_noise","Dark_current_noise", "Additional_background_noise","Sky_noise", "CIC_noise", "RN_final","Signal_resolution","Signal_el","sky","CIC_charge","Dark_current_f","RN","Additional_background"]:
                setattr(self, name, getattr(self,name)*np.ones(n))
        self.factor = self.factor*np.ones(n) if type(self.factor)== np.float64 else self.factor
        self.noises = np.array([self.signal_noise*self.factor,  self.Dark_current_noise*self.factor,  self.Sky_noise*self.factor, self.RN_final*self.factor, self.CIC_noise*self.factor, self.Additional_background_noise*self.factor, self.Signal_resolution]).T
        self.electrons_per_pix =  np.array([self.Signal_el,  self.Dark_current_f,  self.sky,  self.RN_final, self.CIC_charge, self.Additional_background]).T
        self.names = ["Signal","Dark current", "Sky", "Read noise","CIC", "Extra background"]
        self.snrs=self.Signal_resolution /self.Total_noise_final

        if np.ndim(self.noises)==2:
            self.percents =  100* np.array(self.noises).T[:-1,:]**2/self.Total_noise_final**2
        else:
            self.percents =  100* np.array(self.noises).T[:-1]**2/self.Total_noise_final**2            
        
        self.el_per_pix = self.Signal_el + self.sky + self.CIC_charge +  self.Dark_current_f
        n_sigma = 5
        self.signal_nsig_e_resol_nframe = (n_sigma**2 * self.ENF + n_sigma**2 * np.sqrt(4*self.Total_noise_final**2 - 4*self.signal_noise_nframe**2 + self.ENF**2*n_sigma**2))/2
        self.eresolnframe2lu = self.Signal_LU/self.Signal_resolution
        self.signal_nsig_LU = self.signal_nsig_e_resol_nframe * self.eresolnframe2lu
        self.signal_nsig_ergs = convert_LU2ergs(self.signal_nsig_LU, self.wavelength) # self.signal_nsig_LU * self.lu2ergs
        self.extended_source_5s = self.signal_nsig_ergs * (self.pixel_scale*self.PSF_RMS_det)**2
        self.point_source_5s = self.extended_source_5s * 1.30e57
        self.time2reach_n_sigma_SNR = self.acquisition_time *  np.square(n_sigma / self.snrs)
        # print("factor=",self.factor[self.i])
        # print("N_images_true=",np.sqrt(self.N_images_true)[self.i] )
        # print("resolution_element=", self.resolution_element)
        # print("N_resol_element_A=",np.sqrt(self.N_resol_element_A))
        # print("lambda_stack=",self.lambda_stack)
        # print("dispersion=",self.dispersion)
        # print("cosmic_ray_loss=",np.sqrt(self.cosmic_ray_loss)[self.i])
        # print("N_images=",np.sqrt(self.N_images)[self.i])

        #TODO change this ratio of 1.30e57
        # from astropy.cosmology import Planck15 as cosmo
        # 4*np.pi* (cosmo.luminosity_distance(z=0.7).to("cm").value)**2 = 2.30e57


       

    def PlotNoise(self,title='',x='exposure_time', lw=8):
        """
        Generate a plot of the evolution of the noise budget with one parameter:
        exposure_time, Sky_CU, acquisition_time, Signal, EM_gain, RN, CIC_charge, Dard_current, readout_time, smearing, temperature, PSF_RMS_det, PSF_RMS_mask, QE, extra_background, cosmic_ray_loss_per_sec
        """
        fig, axes= plt.subplots(4, 1, figsize=(12, 8), sharex=True) # fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(12, 7), sharex=True) #figsize=(9, 5.5)
        ax1, ax2,ax3, ax4  = axes
        labels = ['%s: %0.3f (%0.1f%%)'%(name,self.electrons_per_pix[self.i,j],100*self.electrons_per_pix[self.i,j]/np.nansum(self.electrons_per_pix[self.i,:])) for j,name in enumerate(self.names)]

        # ax1 
        for i,(name,c) in enumerate(zip(self.names,self.colors)):
            ax1.plot(getattr(self,x), self.noises[:,i]/self.factor,label='%s: %0.2f (%0.1f%%)'%(name,self.noises[self.i,i]/self.factor[self.i],self.percents[i,self.i]),lw=lw,alpha=0.8,c=c)
        ax1.plot(getattr(self,x), np.nansum(self.noises[:,:-1],axis=1)/self.factor,label='%s: %0.2f (%0.1f%%)'%("Total",np.nansum(self.noises[self.i,-1])/self.factor[self.i],np.nansum(self.percents[:,self.i])),lw=lw,alpha=0.4,c="k")
        ax1.legend(loc='upper right')
        ax1.set_ylabel('Noise (e-/pix/exp)')

        # ax1b = ax1.secondary_yaxis("right", functions=( lambda x:  x * self.factor[self.i], lambda x:x / self.factor[self.i] ))
        # self.ax1b = ax1b
        # ax1b.set_ylabel("Noise (e-/res/N frames)")#r"%0.1f,%0.1f,%0.1f"%(self.factor[self.i],self.resolution_element , np.sqrt(self.N_resol_element_A)))


        # ax2 
        ax2.grid(False)
        self.stackplot1 = ax2.stackplot(getattr(self,x),  np.array(self.electrons_per_pix).T[:,:],alpha=0.7,colors=self.colors,labels=labels)
        ax2.set_ylabel('e-/pix/frame')
        ax2.legend(loc='upper right',title="Overall background: %0.3f (%0.1f%%)"%(np.nansum(self.electrons_per_pix[self.i,1:]),100*np.nansum(self.electrons_per_pix[self.i,1:])/np.nansum(self.electrons_per_pix[self.i,:])))
        ax2.set_xlim((getattr(self,x).min(),getattr(self,x).max()))


        # ax2b = ax2.secondary_yaxis("right", functions=( lambda x:  x * self.factor[self.i]**2, lambda x:x / self.factor[self.i]**2 ))
        # self.ax2b = ax2b
        # ax2b.set_ylabel(r"%0.1f,%0.1f,%0.1f"%(self.factor[self.i],self.resolution_element , np.sqrt(self.N_resol_element_A)))



        # ax3
        ax3.grid(False)
        self.stackplot2 = ax3.stackplot(getattr(self,x), self.snrs * np.array(self.noises).T[:-1,:]**2/self.Total_noise_final**2,alpha=0.7,colors=self.colors)
        ax3.set_ylim((0,np.nanmax(self.SNR)))
        ax3.set_ylabel('SNR (res, N frames)')        

        # ax3b = ax3.secondary_yaxis("right", functions=( lambda x: x / self.factor[self.i]**2, lambda x: x * self.factor[self.i]**2))
        # self.ax3b = ax3b
        # ax3b.set_ylabel(r" SNR(res/N frames")



        # ax4
        ax4.plot(getattr(self,x), np.log10(self.extended_source_5s),"-",lw=lw-1,label="SNR=5 Flux/Pow on one elem resolution (%0.2f-%0.2f)"%(np.log10(self.point_source_5s[self.i]),np.nanmin(np.log10(self.point_source_5s))),c="k")
        # if self.instrument==FIREBall:
        if "FIREBall" in self.instrument:

            ax4.plot(getattr(self,x), np.log10(self.extended_source_5s/np.sqrt(2)),"-",lw=lw-1,label="Two elem resolution (%0.2f-%0.2f)"%(np.log10(self.point_source_5s[self.i]/np.sqrt(2)),np.nanmin(np.log10(self.point_source_5s/np.sqrt(2)))),c="grey")
            # ax4.plot(getattr(self,x), np.log10(self.extended_source_5s/np.sqrt(40)),"-",lw=lw-1,label="20 sources stacked on 2 res elem. (%0.2f-%0.2f)"%(np.log10(self.point_source_5s[self.i]/np.sqrt(40)),np.nanmin(np.log10(self.point_source_5s/np.sqrt(40)))),c="lightgrey")
            # ax4.plot(getattr(self,x), np.log10(self.extended_source_5s/np.sqrt(2)/30),"-",lw=lw-1,label="Sources transported to high z: (%0.2f-%0.2f) \ngain of factor 22-50 depending on line resolution"%(np.log10(self.point_source_5s[self.i]/np.sqrt(2)/30),np.nanmin(np.log10(self.point_source_5s/np.sqrt(2)/30))),c="whitesmoke")
        T2 =  lambda x:np.log10(10**x/1.30e57)
        self.pow_2018 = 42.95
        self.pow_best = 41.74
        ax4b = ax4.secondary_yaxis("right", functions=(lambda x:np.log10(10**x * 1.30e57),T2))
        if ("FIREBall" in self.instrument) & (1==0):
            ax4.plot([getattr(self,x).min(),getattr(self,x).min(),np.nan,getattr(self,x).max(),getattr(self,x).max()],[T2(self.pow_2018),T2(self.pow_best),np.nan,T2(self.pow_2018),T2(self.pow_best)],lw=lw,label="2018 flight (%0.1f) - most optimistic case (%0.1f)"%(self.pow_2018,self.pow_best),c="r",alpha=0.5)
        self.T2=T2
        self.ax4b = ax4b
        ax4.legend(loc="upper right", fontsize=8,title="Left: Extend. source F, Right: Point source power" )
        ax4.set_ylabel(r"Log(erg/cm$^2$/s/asec$^2$)")
        ax4b.set_ylabel(r" Log(erg/s)")

        axes[-1].set_xlabel(x)
        ax1.tick_params(labelright=True,right=True)
        ax2.tick_params(labelright=True,right=True)
        ax3.tick_params(labelright=True,right=True)
        fig.tight_layout(h_pad=0.01)
        return fig 

    


    def interpolate_optimal_threshold(self,flux = 0.1,dark_cic_sky_noise=None,plot_=False,title='',i=0):
        """
        Return the threshold optimizing the SNR
        """
        #self.Signal_el if np.isscalar(self.Signal_el) else 0.3
        EM_gain = self.EM_gain #if np.isscalar(self.EM_gain) else self.EM_gain[i]
        RN= self.RN #if np.isscalar(self.RN) else self.RN[i]#80
        CIC_noise = self.CIC_noise #if np.isscalar(self.CIC_noise) else self.CIC_noise[i]
        dark_noise = self.Dark_current_noise #if np.isscalar(self.Dark_current_noise) else self.Dark_current_noise[i]
         
        try:
            Sky_noise = self.Sky_noise_pre_thresholding #if np.isscalar(self.Sky_noise_pre_thresholding) else self.Sky_noise_pre_thresholding[i]
        except AttributeError:
            raise AttributeError('You must use counting_mode=True to use interpolate_optimal_threshold method.')

        noise_value = CIC_noise**2+dark_noise**2+Sky_noise**2
        
        gains=np.linspace(800,2500,n)#self.len_xaxis)
        rons=np.linspace(30,120,n)#self.len_xaxis)
        fluxes=np.linspace(0.01,0.7,n)#self.len_xaxis)
        smearings=np.linspace(0,2,n)#self.len_xaxis)
        noise=np.linspace(0.002,0.05,n)#self.len_xaxis)
        if (n==6)|(n==10):
            coords = (gains, rons, fluxes, smearings)
            point = (EM_gain, RN, flux, self.smearing)            
        elif n==5:
            coords = (gains, rons, fluxes, smearings,noise)
            point = (EM_gain, RN, flux, self.smearing,noise_value)
        else:
            print(n,EM_gain, RN, flux, self.smearing,noise_value)
            
        if ~np.isscalar(noise_value) |  ~np.isscalar(self.smearing) | ~np.isscalar(EM_gain) | ~np.isscalar(RN):
            point = np.repeat(np.zeros((4,1)), self.len_xaxis, axis=1).T
            point[:,0] =  self.EM_gain
            point[:,1] = self.RN
            point[:,2] = flux
            point[:,3] = self.smearing
        fraction_rn =interpn(coords, table_fraction_rn, point,bounds_error=False,fill_value=None)
        fraction_signal =interpn(coords, table_fraction_flux, point,bounds_error=False,fill_value=None)
        threshold = interpn(coords, table_threshold, point,bounds_error=False,fill_value=None)
        snr_ratio = interpn(coords, table_snr, point,bounds_error=False,fill_value=None)

        if type(self.smearing)==float:
            if self.smearing == 0:
                a = Table.read("fraction_flux.csv")
                threshold = 5.5
                fraction_signal = np.interp(self.EM_gain/self.RN,a["G/RN"],a["fractionflux"])
            # fraction_rn = f(flux=0.1,EM_gain=self.EM_gain, RN=self.RN)
            # fraction_signal = f2(flux=0.1,EM_gain=self.EM_gain, RN=self.RN)
            # snr_ratio = f3(flux=0.1,EM_gain=self.EM_gain, RN=self.RN)

        return threshold, fraction_signal, fraction_rn, snr_ratio#np.nanmax(SNR1/SNR_analogic)
 
    
 

    def compute_optimal_threshold(self,flux = 0.1,dark_cic_sky_noise=None,plot_=False,title='',i=0,axes=None,size= (int(1e3),int(1e3)),size_bin=25, threshold=-1000):
        """ 
        Create a ADU value histogram and defin the threshold so that it gives the optimal SNR based on RN, smearing, noise, flux, gain
        Function is pretty slow so output of this function has been saved and can then directly be used with interpolation (see function interpolate_optimal_threshold)
        """
        #self.Signal_el if np.isscalar(self.Signal_el) else 0.3
        EM_gain = self.EM_gain if np.isscalar(self.EM_gain) else self.EM_gain[i]#1000
        RN = self.RN if np.isscalar(self.RN) else self.RN[i]#80
        CIC_noise = self.CIC_noise if np.isscalar(self.CIC_noise) else self.CIC_noise[i]
        dark_noise = self.Dark_current_noise if np.isscalar(self.Dark_current_noise) else self.Dark_current_noise[i]
        try:
            Sky_noise = self.Sky_noise_pre_thresholding if np.isscalar(self.Sky_noise_pre_thresholding) else self.Sky_noise_pre_thresholding[i]
        except AttributeError:
            raise AttributeError('You must use counting_mode=True to use compute_optimal_threshold method.')

        im = np.random.poisson(flux, size=size)
        values,bins = np.histogram(im,bins=[-0.5,0.5,1.5,2.5])
        ConversionGain=1#/4.5
        imaADU = np.random.gamma(im, EM_gain) *ConversionGain
        bins = np.arange(np.min(imaADU)-5*RN*ConversionGain,np.max(imaADU)+5*RN*ConversionGain,25)
        # bins = np.linspace(-500,10000,400)
        #imaADU = (np.random.gamma(im, EM_gain) + np.random.normal(0, RN, size=size))*ConversionGain
        imaADU_copy = imaADU.copy()
        imaADU_copy += np.random.normal(0, RN, size=size)*ConversionGain

        if plot_:
            if axes is None:
                fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(12, 7))#,figsize=(9,5))
            else:
                fig=0
                ax1, ax2 = axes
                ax1.clear()
                ax2.clear()
            val0,_,l0 = ax1.hist(imaADU_copy[im==0],bins=bins,alpha=0.5,log=True,histtype='step',lw=0.7,color='k',label='Before ampl & smearing')
            val1,_,l1 = ax1.hist(imaADU_copy[im==1],bins=bins,alpha=0.5,log=True,histtype='step',lw=0.7,color='k')
            val2,_,l2 = ax1.hist(imaADU_copy[im==2],bins=bins,alpha=0.5,log=True,histtype='step',lw=0.5,color='k')
            # val3,_,l3 = ax1.hist(imaADU[im==3],bins=bins,alpha=0.5,log=True,histtype='step',lw=0.5,color='k')
            # val4,_,l4 = ax1.hist(imaADU[im==4],bins=bins,alpha=0.5,log=True,histtype='step',lw=0.5,color='k')
            # val5,_,l5 = ax1.hist(imaADU[im==5],bins=bins,alpha=0.5,log=True,histtype='step',lw=0.5,color='k')


        if self.smearing > 0:
            # print(SmearExpDecrement)
            smearing_kernels = variable_smearing_kernels(
                imaADU, self.smearing, SmearExpDecrement=5e4)
            offsets = np.arange(6)
            A = dia_matrix(
                (smearing_kernels.reshape((6, -1)), offsets),
                shape=(imaADU.size, imaADU.size))

            imaADU = A.dot(imaADU.ravel()).reshape(imaADU.shape)
        imaADU += np.random.normal(0, RN, size=size)*ConversionGain

        if 1==1:
            b = (bins[:-1]+bins[1:])/2
            above_threshold =np.array([np.sum(val1[b>bi]) for bi in b])/np.sum(val1)
            rn_frac = np.array([np.sum(val0[b>bi]) for bi in b])/  np.array([np.sum(val1[b>bi]) for bi in b])#   np.sum(val0) 
            rn_frac = np.array([np.sum(val0[b>bi]) for bi in b])/  (np.array([np.sum(val1[b>bi])for bi in b])+np.array([np.sum(val0[b>bi]) for bi in b]))
            rn_noise =  rn_frac #* (RN/( ConversionGain)) # #/(EM_gain*ConversionGain)#/(EM_gain*ConversionGain)
            # rn_noise = RN * np.array([np.sum(val0[b>bi]) for bi in b])/np.sum(val0) #/(EM_gain*ConversionGain)#/(EM_gain*ConversionGain)
            signal12 = flux * np.array([np.sum(val1[b>bi])+np.sum(val2[b>bi]) for bi in b])/(np.sum(val1)+np.sum(val2))
            
            signal1 = flux * above_threshold

            pc = np.ones(len(b))# 
                # ([np.sum(val1[b>bi])for bi in b]/(np.array([np.sum(val1[b>bi])for bi in b])+np.array([np.sum(val0[b>bi]) for bi in b])))
            pc =  ([np.sum(val1[b>bi])for bi in b]/(np.array([np.sum(val1[b>bi])for bi in b])+np.array([np.sum(val0[b>bi]) for bi in b])))

            if dark_cic_sky_noise is None:
                noise = (CIC_noise**2+dark_noise**2+Sky_noise**2 +flux)
            else:
                noise = (dark_cic_sky_noise**2 +flux)  
            print('noises = ',noise)
            denom = np.sqrt(noise* above_threshold*2+np.array(rn_noise)**2)#
            SNR1 = signal1/denom
            print('SNR1 = ',SNR1)
            SNR12 = signal12/ denom
            SNR_analogic = flux/np.sqrt(2*flux+2*noise+(RN/(EM_gain * ConversionGain))**2)
            # print('SNR_analogic = ',SNR_analogic)
            threshold_55 = 5.5*RN*ConversionGain
            id_55 =  np.argmin(abs(threshold_55 - b))
            # if threshold<-5:
            #     id_t = np.nanargmax(SNR1)
            #     threshold = b[id_t]
            # else:
            #     threshold *= RN*ConversionGain
            #     id_t = np.argmin(abs(threshold - b))
            # # print(threshold)
            # fraction_signal = np.sum(val1[id_t:])/np.sum(val1)
            # fraction_rn = np.sum(val0[id_t:])/np.sum(val0)
            lw=3
            if plot_:
                ax2.plot(b,rn_frac,lw=lw,ls=":",c="C0")#,label='RN(RN>T):  %0.2f%% ➛ %0.2f%%'%(100*rn_frac[id_55],100*rn_frac[id_t])
                ax2.plot(b,signal1/flux,lw=lw,ls=":",c="C1")#,label='Signal(Signal>T):  %0.1f%% ➛ %0.1f%%'%(100*signal1[id_55]/flux,100*signal1[id_t]/flux)
                # ax2.plot(b,np.array(rn_noise)**2,label='(RN(RN>T)/EM_gain)**2',lw=lw)
                # ax2.plot(b,pc,lw=lw,ls=":",c="C2")#,label='Fraction(T) of true positive: %0.1f%% ➛ %0.1f%%'%(100*pc[id_55],100*pc[id_t])
                #ax2.plot(b,SNR1/pc,label='SNR without fraction')

                # ax2.plot(b,SNR1/np.nanmax(SNR1),lw=lw,c="C4") # ,label='SNR1: %0.2f%% ➛ %0.2f%%'%(SNR1[id_55],SNR1[id_t])#'%(100*np.sum(val0[id_t:])/np.sum(val0),100*np.sum(val1[id_t:])/np.sum(val1)),lw=lw)
                # ax2.plot(b,SNR12,':',label='SNR12, [N1+N2]/[N0] = %0.2f, frac(N1+N2)=%i%%'%((val1[np.nanargmax(SNR12)]+val2[np.nanargmax(SNR12)])/val0[np.nanargmax(SNR12)],100*np.sum(val1[np.nanargmax(SNR12):]+val2[np.nanargmax(SNR12):])/(np.sum(val1)+np.sum(val2))),lw=lw)
                ax2.plot(b,SNR1/SNR_analogic,lw=lw,ls=":",c="C3")#,label='SNR1 PC / SNR analogic: %0.2f ➛ %0.2f'%(SNR1[id_55]/SNR_analogic,SNR1[id_t]/SNR_analogic)

        if plot_:
            val0,_,l0 = ax1.hist(imaADU[im==0],bins=bins,alpha=0.5,label='0',log=True)
            val1,_,l1 = ax1.hist(imaADU[im==1],bins=bins,alpha=0.5,label='1',log=True)
            val2,_,l2 = ax1.hist(imaADU[im==2],bins=bins,alpha=0.5,label='2',log=True)
            # val3,_,l3 = ax1.hist(imaADU[im==3],bins=bins,alpha=0.5,label='3',log=True)
            # val4,_,l4 = ax1.hist(imaADU[im==4],bins=bins,alpha=0.5,label='4',log=True)
            # val5,_,l5 = ax1.hist(imaADU[im==5],bins=bins,alpha=0.5,label='5',log=True)
            ax1.hist(imaADU.flatten(),bins=bins,label='Total histogram',log=True,histtype='step',lw=1,color='k')
            # ax1.fill_between([bins[np.argmin(val0>val1)],bins[np.argmax(val0>val1)]],[val0.max(),val0.max()],[1.2*val0.max(),1.2*val0.max()],alpha=0.3,color="C0")
            # a0 = np.where(val0>val1)[0].max()
            # a1 = np.where((val1>val2)&(val1>val0))[0].max()
            # a2 = np.where((val2>val3)&(val2>val1))[0].max()
            # a3 = np.where((val3>val4)&(val3>val2))[0].max()
            # a4 = np.where((val4>val5)&(val4>val3))[0].max()
            # ax1.fill_between([ bins[0],bins[a0]],[val0.max(),val0.max()],[1.2*val0.max(),1.2*val0.max()],alpha=0.3,color="C0")
            # ax1.fill_between([bins[a0],bins[a1]],[val0.max(),val0.max()],[1.2*val0.max(),1.2*val0.max()],alpha=0.3,color="C1")
            # ax1.fill_between([bins[a1],bins[a2]],[val0.max(),val0.max()],[1.2*val0.max(),1.2*val0.max()],alpha=0.3,color="C2")
            # ax1.fill_between([bins[a2],bins[a3]],[val0.max(),val0.max()],[1.2*val0.max(),1.2*val0.max()],alpha=0.3,color="C3")
            # ax1.fill_between([bins[a3],bins[a4]],[val0.max(),val0.max()],[1.2*val0.max(),1.2*val0.max()],alpha=0.3,color="C4")
            # ax1.fill_between([bins[a4],bins[-1]],[val0.max(),val0.max()],[1.2*val0.max(),1.2*val0.max()],alpha=0.3,color="C5")
        else:
            val0,_ = np.histogram(imaADU[im==0],bins=bins)#,alpha=0.5,label='0',log=True)
            val1,_ = np.histogram(imaADU[im==1],bins=bins)#,alpha=0.5,label='1',log=True)
            val2,_ = np.histogram(imaADU[im==2],bins=bins)#,alpha=0.5,label='2',log=True)
            val3,_ = np.histogram(imaADU[im==3],bins=bins)
            val4,_ = np.histogram(imaADU[im==4],bins=bins)
            val5,_ = np.histogram(imaADU[im==5],bins=bins)

        b = (bins[:-1]+bins[1:])/2
        above_threshold =np.array([np.sum(val1[b>bi]) for bi in b])/np.sum(val1)
        rn_frac = np.array([np.sum(val0[b>bi]) for bi in b])/  np.array([np.sum(val1[b>bi]) for bi in b])#   np.sum(val0) 
        # rn_frac = np.array([np.sum(val0[b>bi]) for bi in b])/  (np.array([np.sum(val1[b>bi])for bi in b])+np.array([np.sum(val0[b>bi]) for bi in b]))
        rn_noise =  rn_frac #* (RN/( ConversionGain)) # #/(EM_gain*ConversionGain)#/(EM_gain*ConversionGain)
        # rn_noise = RN * np.array([np.sum(val0[b>bi]) for bi in b])/np.sum(val0) #/(EM_gain*ConversionGain)#/(EM_gain*ConversionGain)
        signal12 = flux * np.array([np.sum(val1[b>bi])+np.sum(val2[b>bi]) for bi in b])/(np.sum(val1)+np.sum(val2))
        
        signal1 = flux * above_threshold

        pc = np.ones(len(b))# 
              # ([np.sum(val1[b>bi])for bi in b]/(np.array([np.sum(val1[b>bi])for bi in b])+np.array([np.sum(val0[b>bi]) for bi in b])))
        pc =  ([np.sum(val1[b>bi])for bi in b]/(np.array([np.sum(val1[b>bi])for bi in b])+np.array([np.sum(val0[b>bi]) for bi in b])))
        ENF=2

        if dark_cic_sky_noise is None:
            noise = (CIC_noise**2+dark_noise**2+Sky_noise**2 +flux)
        else:
            noise = (dark_cic_sky_noise**2 +flux)  
        print('noises = ',noise)
        denom = np.sqrt(noise* above_threshold*1+np.array(rn_noise)**2)#
        SNR1 = signal1/denom
        print('SNR1 = ',SNR1)
        SNR12 = signal12/ denom
        SNR_analogic = flux/np.sqrt(ENF*flux+ENF*noise+(RN/(EM_gain * ConversionGain))**2)
        # print('SNR_analogic = ',SNR_analogic)
        threshold_55 = 5.5*RN*ConversionGain
        id_55 =  np.nanargmin(abs(threshold_55 - b))
        if threshold<-5:
            id_t = np.nanargmax(SNR1)
            threshold = b[id_t]
        else:
            threshold *= RN*ConversionGain
            id_t = np.nanargmin(abs(threshold - b))
        # print(threshold)
        fraction_signal = np.nansum(val1[id_t:])/np.nansum(val1)
        fraction_rn = np.nansum(val0[id_t:])/np.nansum(val0)
        lw=3
        if plot_:
            ax2.plot(b,rn_frac,label='RN(RN>T):  %0.2f%% ➛ %0.2f%%'%(100*rn_frac[id_55],100*rn_frac[id_t]),lw=lw,c="C0")
            # ax2.plot(b,denom,label='RN(RN>T):  %0.2f%% ➛ %0.2f%%'%(100*rn_frac[id_55],100*rn_frac[id_t]),lw=lw,c="C0")
            ax2.plot(b,signal1/flux,label='Signal(Signal>T):  %0.1f%% ➛ %0.1f%%'%(100*signal1[id_55]/flux,100*signal1[id_t]/flux),lw=lw,c="C1")
            # ax2.plot(b,pc,label='Fraction(T) of true positive: %0.1f%% ➛ %0.1f%%'%(100*pc[id_55],100*pc[id_t]),lw=lw,c="C2")
            ax2.plot(b,SNR1/SNR_analogic,label='SNR1 PC / SNR analogic: %0.2f ➛ %0.2f'%(SNR1[id_55]/SNR_analogic,SNR1[id_t]/SNR_analogic),lw=lw,c="C3")
            # ax2.plot(b,SNR12/SNR_analogic,label='SNR1 PC / SNR analogic: %0.2f ➛ %0.2f'%(SNR1[id_55]/SNR_analogic,SNR1[id_t]/SNR_analogic),lw=lw,c="C3")
            ax2.axhline(y=1.41,ls=":",c="k")
            # ax2.plot(b,signal1/denom,label="signal1/denom",lw=lw,c="C4")
            # ax2.plot(b,signal12/denom,label="signal1/denom",lw=lw,c="C4")
            # ax2.plot(b,SNR1/SNR_analogic,label='SNR1 PC / SNR analogic: %0.2f ➛ %0.2f'%(SNR1[id_55]/SNR_analogic,SNR1[id_t]/SNR_analogic),lw=lw,c="C3")
            # ax2.plot(b,np.array(rn_noise)**2,label='(RN(RN>T)/EM_gain)**2',lw=lw)
            #ax2.plot(b,SNR1/pc,label='SNR without fraction')
            # print(SNR1)
            # print(SNR1/SNR1.max())
            # ax2.plot(b,SNR1/np.nanmax(SNR1),label='SNR1: %0.2f%% ➛ %0.2f%%'%(SNR1[id_55],SNR1[id_t]),lw=lw,c="C4") #'%(100*np.sum(val0[id_t:])/np.sum(val0),100*np.sum(val1[id_t:])/np.sum(val1)),lw=lw)
            # ax2.plot(b,SNR12,':',label='SNR12, [N1+N2]/[N0] = %0.2f, frac(N1+N2)=%i%%'%((val1[np.nanargmax(SNR12)]+val2[np.nanargmax(SNR12)])/val0[np.nanargmax(SNR12)],100*np.sum(val1[np.nanargmax(SNR12):]+val2[np.nanargmax(SNR12):])/(np.sum(val1)+np.sum(val2))),lw=lw)



            # ax2.plot(b,SNR12/SNR_analogic,':',label='SNR12 PC / SNR analogic',lw=lw)
            # ax2.set_yscale('log')
            ax2.set_ylim(ymin=1e-5)
            
            # ax2.plot(b,SNR1,label='[N1]/[N0] = %0.2f, frac(N1)=%i%%'%(val1[id_t]/val0[id_t],100*np.sum(val1[id_t:])/np.sum(val1)))
            # ax2.plot(b,SNR12,label='[N1+N2]/[N0] = %0.2f, frac(N1+N2)=%i%%'%((val1[np.nanargmax(SNR12)]+val2[np.nanargmax(SNR12)])/val0[np.nanargmax(SNR12)],100*np.sum(val1[np.nanargmax(SNR12):]+val2[np.nanargmax(SNR12):])/(np.sum(val1)+np.sum(val2))))

            ax2.legend(title = "T = 5.5σ ➛ %0.1fσ "%(threshold/(RN*ConversionGain)), fontsize=10)
            ax2.set_xlabel('ADU')
            ax1.set_ylabel('#')
            ax2.set_ylabel('SNR')
            ax1.plot([threshold,threshold],[0,np.max(val0)],':',c='k',label=r"SNR optimal threshold")
            ax2.plot([threshold,threshold],[0,1],':',c='k')
            ax1.plot([threshold_55,threshold_55],[0,np.max(val0)],'-.',c='k',label=r"5.5 $\sigma_{RN}$ threshold")
            ax2.plot([threshold_55,threshold_55],[0,1],'-.',c='k')
            L = ax1.legend(fontsize=10)
            L.get_texts()[1].set_text('0 e- : %i%%, fraction kept: %0.2f%%'%(100*values[0]/(size[0]*size[1]),100*np.sum(val0[id_t:])/np.sum(val0)))
            L.get_texts()[2].set_text('1 e- : %i%%, fraction kept: %0.2f%%'%(100*values[1]/(size[0]*size[1]),100*np.sum(val1[id_t:])/np.sum(val1)))
            L.get_texts()[3].set_text('2 e- : %i%%, fraction kept: %0.2f%%'%(100*values[2]/(size[0]*size[1]),100*np.sum(val2[id_t:])/np.sum(val2)))

            ax1.set_title(title+'Gain = %i, RN = %i, flux = %0.2f, smearing=%0.1f, Threshold = %i = %0.2f$\sigma$'%(EM_gain,RN,flux,self.smearing, threshold,threshold/(RN*ConversionGain)))
            ax1.set_xlim(xmin=bins.min(),xmax=5000)#bins.max())
            # ax1.set_xlim(xmin=bins.min(),xmax=bins.max()/2)
            ax2.set_yscale("symlog")
            if axes is None:
                fig.tight_layout()
            plt.show()
            # return fig
        # sys.exit()
        return threshold/(RN*ConversionGain), fraction_signal, fraction_rn, np.nanmax(SNR1/SNR_analogic)
    

Observation(exposure_time=np.linspace(50,1500,50),RN=40,smearing=1.5,counting_mode=True,plot_=False).compute_optimal_threshold(plot_=True,flux=0.1,dark_cic_sky_noise=1,size= (int(3e2),int(1e3)))
