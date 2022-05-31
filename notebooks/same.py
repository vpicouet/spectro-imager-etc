#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:35:03 2022

@author: Vincent
"""

from ipywidgets import Button, Layout, jslink, IntText, IntSlider, interactive, interact, HBox, Layout, VBox
# %matplotlib widget
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
# plt.style.use('dark_background')
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
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

#fixed parameters
QE = 0.55
resolution_element=57#microns
pixel_size = 13#microns
# Gain_ADU = 0.53 #e-/e-
Throughput = 0.13
Atmosphere = 0.5
colors = ['#E24A33','#348ABD','#988ED5','#FBC15E','#8EBA42','#FFB5B8','#777777']

n=6
table_threshold = fits.open("threshold_%s.fits"%(n))[0].data
table_snr = fits.open("snr_max_%s.fits"%(n))[0].data
table_fraction_rn = fits.open("fraction_rn_%s.fits"%(n))[0].data
table_fraction_flux = fits.open("fraction_flux_%s.fits"%(n))[0].data
# print(table_threshold.shape)


def variable_smearing_kernels(image, Smearing=1.5, SmearExpDecrement=50000):
    """Creates variable smearing kernels for inversion
    """
    import numpy as np
    
    smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
    smearing_kernels = np.exp(-np.arange(6)[:, np.newaxis, np.newaxis] / smearing_length)
    smearing_kernels /= smearing_kernels.sum(axis=0)
    return smearing_kernels   


class Observation:
    @initializer
    def __init__(self, exposure_time=50, counting_mode=False, Signal=1.25E-17, EM_gain=1400, RN=109, CIC_charge=0.005, Dard_current=0.08, Sky_LU=10000, readout_time=1.5, flight_background_damping = 0.9, Additional_background_2018 = 0,acquisition_time = 2,smearing=0,i=0,plot_=False,temperature=-100):#,photon_kept=0.7
        self.ENF = 2 if counting_mode else 1
        self.CIC_noise = np.sqrt(CIC_charge) if counting_mode else np.sqrt(CIC_charge*2)
        self.Dark_current_f = Dard_current * exposure_time / 3600 # e/pix/frame
        self.Dark_current_noise =  np.sqrt(self.Dark_current_f) if counting_mode else np.sqrt(self.Dark_current_f*2)
        
        self.lu2ergs = 2.33E-19/1000        
        self.Sky_ = Sky_LU*self.lu2ergs#/1000*2.33E-19 # ergs/cm2/s/arcsec^2 
        
        
        # self.temperature=-100
        if counting_mode:
            #for now we put the regular QE without taking into account the photon fraciton, because then infinite loop
            self.factor_el = QE * Throughput * Atmosphere*(1.1*np.pi/180/3600)**2*np.pi*100**2/4
            self.sky = Sky_LU*self.factor_el*exposure_time  # el/pix/frame
            self.Sky_f =  self.sky * EM_gain #* Gain_ADU  # el/pix/frame
            self.Sky_noise_pre_thresholding = np.sqrt(self.sky) if counting_mode else np.sqrt(self.sky*2)
                #self.Signal_LU = Signal / self.lu2ergs# LU(self.Sky_/self.Sky_LU)#ergs/cm2/s/arcsec^2 
            #self.Signal_el = self.Signal_LU*self.factor_el*exposure_time 

            #             self.Photon_count_loss = 0.7 if readout_time==1.5 else 0.8 #a contraindre...
            self.n_threshold, self.Photon_fraction_kept, self.RN_fraction_kept, self.gain_thresholding, self.true_positive, self.fake_negative = self.compute_optimal_threshold(plot_=plot_, i=i) #photon_kept
        # if counting_mode:
            # self.n_threshold, self.Photon_fraction_kept, self.RN_fraction_kept, self.gain_thresholding = self.interpolate_optimal_threshold(plot_=plot_, i=i)
            # print(self.Photon_fraction_kept.min(), self.Photon_fraction_kept.max())
            # print(self.RN_fraction_kept.min(), self.RN_fraction_kept.max())
            # print(self.gain_thresholding.min(), self.gain_thresholding.max())
            # print(self.RN_fraction_kept, self.gain_thresholding)
        else:
            self.n_threshold, self.Photon_fraction_kept, self.RN_fraction_kept, self.gain_thresholding = np.zeros(50),np.ones(50),np.ones(50), np.zeros(50) #0,1,1, 0
            # self.n_threshold, self.Photon_fraction_kept, self.RN_fraction_kept, self.gain_thresholding = self.compute_optimal_threshold(plot_=plot_, i=i) #photon_kept

        self.cosmic_ray_loss = np.minimum(0.005*(exposure_time+readout_time/2),1)
        self.QE_efficiency = self.Photon_fraction_kept * QE#* (1-cosmic_ray_loss)#ne devrait pas etre la 
        # if type(self.QE_efficiency)!= float:
        #     print(self.QE_efficiency.min(), self.QE_efficiency.max())

        #for now we put the regular QE without taking into account the photon fraciton, because then infinite loop
        self.factor_el = self.QE_efficiency * Throughput * Atmosphere*(1.1*np.pi/180/3600)**2*np.pi*100**2/4
        self.sky = Sky_LU*self.factor_el*exposure_time  # el/pix/frame
        self.Sky_f =  self.sky * EM_gain #* Gain_ADU  # ADU/pix/frame
        self.Sky_noise = np.sqrt(self.sky) if counting_mode else np.sqrt(self.sky*2)
            

        #self.RN_fraction_kept = 0.05 if counting_mode else 1
        self.RN_final = RN  * self.RN_fraction_kept / EM_gain #Are we sure about that? we should a
 

        self.Additional_background = Additional_background_2018 * exposure_time *(1-flight_background_damping) # e/pix/f
        self.Additional_background_noise = np.sqrt(self.Additional_background) if counting_mode else np.sqrt(self.Additional_background*2)



        
        
        self.N_images = acquisition_time*3600/(exposure_time+readout_time)
        coeff_stack = 1 #TBC, why was it set to 2
        self.N_images_true = self.N_images * coeff_stack * (1-self.cosmic_ray_loss)
        self.Total_sky = self.N_images_true * self.sky
        self.sky_resolution = self.Total_sky * (resolution_element/pixel_size)**2# el/N exposure/resol
        self.Signal_LU = Signal / self.lu2ergs# LU(self.Sky_/self.Sky_LU)#ergs/cm2/s/arcsec^2 
        self.Signal_el = self.Signal_LU*self.factor_el*exposure_time  # el/pix/frame#     Signal * (sky / Sky_)  #el/pix
        # print(Signal )
        # print(self.Signal_LU )
        # print(self.Signal_el )
#         if counting_mode:
#             print('%0.1f < ExpTime < %0.1f' %(0.01/self.factor_el/self.Signal_LU,0.1/self.factor_el/self.Signal_LU))
    
        self.Signal_resolution = self.Signal_el *self.N_images_true* (resolution_element/pixel_size)**2# el/N exposure/resol
        self.eresolnframe2lu = self.Signal_LU/self.Signal_resolution
        self.signal_noise = np.sqrt(self.Signal_el) if counting_mode else np.sqrt(self.Signal_el*2)     #el / resol/ N frame
        self.signal_noise_resol = self.signal_noise *resolution_element/pixel_size   # el/resol/frame
        self.signal_noise_nframe = self.signal_noise *np.sqrt(self.N_images_true)  # el/resol/frame
        self.Total_noise_final = np.sqrt(self.signal_noise**2 + self.Dark_current_noise**2  + self.Additional_background_noise**2 + self.Sky_noise**2 + self.CIC_noise**2 + self.RN_final**2   ) #e/  pix/frame
        self.factor = np.sqrt(self.N_images_true) * (resolution_element/pixel_size)
        self.Total_noise_nframe = self.Total_noise_final * np.sqrt(self.N_images_true)
        self.Total_noise_resol = self.Total_noise_nframe * (resolution_element/pixel_size)
        self.SNR = self.Signal_resolution/self.Total_noise_resol
        self.Total_noise_final = self.factor*np.sqrt(self.signal_noise**2 + self.Dark_current_noise**2  + self.Additional_background_noise**2 + self.Sky_noise**2 + self.CIC_noise**2 + self.RN_final**2   ) #e/  pix/frame
        if type(self.Total_noise_final + self.Signal_resolution) == np.float64:#to correct
            n=0
        else:
            n =len(self.Total_noise_final + self.Signal_resolution) 
        if n>1:
            for name in ["signal_noise","Dark_current_noise", "Additional_background_noise","Sky_noise", "CIC_noise", "RN_final","Signal_resolution","Signal_el","sky","CIC_charge","Dark_current_f","RN"]:
                setattr(self, name, getattr(self,name)*np.ones(n))
        # self.noises = np.array([self.signal_noise*self.factor,  self.Dark_current_noise*self.factor,  self.Additional_background_noise*self.factor, self.Sky_noise*self.factor, self.CIC_noise*self.factor, self.RN_final*self.factor, self.Signal_resolution]).T
        self.noises = np.array([self.signal_noise*self.factor,  self.Dark_current_noise*self.factor,  self.Sky_noise*self.factor, self.CIC_noise*self.factor, self.RN_final*self.factor, self.Signal_resolution]).T

        # self.electrons_per_pix =  np.array([self.Signal_el,  self.Dark_current_f,  0*self.Additional_background_noise, self.sky, self.CIC_charge, self.RN_final]).T
        self.electrons_per_pix =  np.array([self.Signal_el,  self.Dark_current_f,  self.sky, self.CIC_charge, self.RN_final]).T

        self.snrs=self.Signal_resolution /self.Total_noise_final
        if np.ndim(self.noises)==2:
            self.percents =  100* np.array(self.noises).T[:-1,:]**2/self.Total_noise_final**2
        else:
            self.percents =  100* np.array(self.noises).T[:-1]**2/self.Total_noise_final**2            
        self.el_per_pix = self.Signal_el + self.sky + CIC_charge +  self.Dark_current_f

        # self.percents =  100* np.array(self.noises).T**2/self.Total_noise_final**2
       

    def PlotNoise(self,title='',x='exposure_time'):
        # plt.style.use('seaborn')
        lw=8
        fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        # fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True)        # print(self.i)
        for i,(name,c) in enumerate(zip(["signal","Dark current","Sky", "CIC", "Read noise"],colors)):#, "Add. background"
            # ax1.plot(getattr(self,x), getattr(self,name)*self.factor,label='%s: %i (%0.1f%%)'%(name,(getattr(self,name)*self.factor)[np.argmax(self.snrs)],self.percents[i,np.argmax(self.snrs)]),lw=lw,alpha=0.7,c=c)
            # ax1.plot(getattr(self,x), getattr(self,name)*self.factor,label='%s: %i (%0.1f%%)'%(name,(getattr(self,name)*self.factor)[self.i],self.percents[i,self.i]),lw=lw,alpha=0.8,c=c)
            ax1.plot(getattr(self,x), self.noises[:,i],label='%s: %i (%0.1f%%)'%(name,self.noises[self.i,i],self.percents[i,self.i]),lw=lw,alpha=0.8,c=c)

        # ax1.plot(getattr(self,x), getattr(self,"Total_noise_final")*self.factor,label="Total_noise_final", lw=lw,alpha=0.7,c='k')
        ax1.legend(loc='upper right')
        # ax3.plot([getattr(self,x)[np.argmax(self.snrs)],getattr(self,x)[np.argmax(self.snrs)]],[0,np.max(self.snrs)],':',c='k')
        # ax2.plot([getattr(self,x)[np.argmax(self.snrs)],getattr(self,x)[np.argmax(self.snrs)]],[0,100],':',c='k')
        # ax3.plot(getattr(self,x), self.snrs,lw=0,c='k',label='$SNR_{max}$=%0.1f at t=%i'%(np.max(self.snrs),getattr(self,x)[np.argmax(self.snrs)]))
        # ax3.legend(loc='upper right')
        ax1.grid(False)
#         ax1.set_yscale('log')
        ax2.grid(False)
        ax3.grid(False)
        ax3.stackplot(getattr(self,x), self.snrs * np.array(self.noises).T[:-1,:]**2/self.Total_noise_final**2,alpha=0.7,colors=colors)
        
        # ax2.stackplot(getattr(self,x),self.percents,alpha=0.7,colors=colors)
        # ax2.set_ylim((0,99.99))
        # ax2.set_ylabel('Noise contribution %')
        # if self.counting_mode:
        #     ax2.stackplot(getattr(self,x),  np.array(self.ADU_per_pix).T[:,:],alpha=0.7,colors=colors)
        #     ax2.set_ylabel('ADU/pix %')
        # else:
        # labels = ['%s: %0.3f (%0.1f%%)'%(name,(getattr(self,"electrons_per_pix")[j])[self.i],1) for j,name in enumerate(["Signal_el","Dark_current_f", "Additional_background_noise","sky", "CIC_charge", "RN_final"])]
        labels = ['%s: %0.3f (%0.1f%%)'%(name,getattr(self,"electrons_per_pix")[self.i,j],100*getattr(self,"electrons_per_pix")[self.i,j]/np.sum(getattr(self,'electrons_per_pix')[self.i,:])) for j,name in enumerate(["Signal","Dark current", "Sky", "CIC", "Read noise"])]#"Add. background",
        ax2.stackplot(getattr(self,x),  np.array(self.electrons_per_pix).T[:,:],alpha=0.7,colors=colors,labels=labels)
        ax2.set_ylabel('e-/pix/frame')
        ax2.legend(loc='upper right')
            # ax2.set_yscale('log')
        # ax2.set_ylim((0,99.99))
# Signal_el,  self.Dark_current_f,  0*self.Additional_background_noise, self.sky, self.CIC_charge, self.RN_final
            # Here we put the number of e- or ADU per pixels? both are interesting as only ADU gives the readnoise contrib.but its in e- that we know if thrsholding if efficient. So could be in e- and when we hit thresholding we put ADU.

        
        # stackplot(time,  np.array(self.noises)[:,-1]**2 /np.array(self.noises).T[:1,:]**2)
        ax2.set_xlim((getattr(self,x).min(),getattr(self,x).max()))
        ax3.set_ylim((0,np.max(self.SNR)))
        # ax1.set_ylim((time.min(),time.max()))
        ax3.set_xlabel(x)
        ax3.set_ylabel('SNR')
        ax1.set_ylabel('Noise (e-/res/N frames)')
        ax1.tick_params(labelright=True,right=True)
        ax2.tick_params(labelright=True,right=True)
        ax3.tick_params(labelright=True,right=True)
        # ax1.set_title('$t_{aqu}$:%0.1fh,G$_{EM}$:%i, Counting:%s - SNR$_{MAX}$=%0.1f'%(new.acquisition_time,new.EM_gain,new.counting_mode,np.max(new.SNR)),y=1)
        # fig.suptitle('pompo,')
        # ax1.set_title(title+'Flux:%s, $t_{aqu}$:%0.1fh, G$_{EM}$:%i, Counting:%s'%(self.Signal,self.acquisition_time,self.EM_gain,self.counting_mode))
        fig.tight_layout()
        return fig 

    
    def compute_optimal_threshold(self,flux = 0.1,dark_cic_sky_noise=None,plot_=False,title='',i=0):
        #self.Signal_el if np.isscalar(self.Signal_el) else 0.3
        Emgain = self.EM_gain if np.isscalar(self.EM_gain) else self.EM_gain[i]#1000
        RN = self.RN if np.isscalar(self.RN) else self.RN[i]#80
        CIC_noise = self.CIC_noise if np.isscalar(self.CIC_noise) else self.CIC_noise[i]
        dark_noise = self.Dark_current_noise if np.isscalar(self.Dark_current_noise) else self.Dark_current_noise[i]
         
        try:
            Sky_noise = self.Sky_noise_pre_thresholding if np.isscalar(self.Sky_noise_pre_thresholding) else self.Sky_noise_pre_thresholding[i]
        except AttributeError:
            raise AttributeError('You must use counting_mode=True to use compute_optimal_threshold method.')

        size= (int(1e3),int(1e3))
        im = np.random.poisson(flux, size=size)
        values,bins = np.histogram(im,bins=[-0.5,0.5,1.5,2.5])
        ConversionGain=1#/4.5
        imaADU = np.random.gamma(im, Emgain) *ConversionGain
        bins = np.arange(np.min(imaADU)-5*RN*ConversionGain,np.max(imaADU)+5*RN*ConversionGain,25)
        # bins = np.linspace(-500,10000,400)
        #imaADU = (np.random.gamma(im, Emgain) + np.random.normal(0, RN, size=size))*ConversionGain
        if plot_:
            fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(9,5))
            val0,_,l0 = ax1.hist(imaADU[im==0],bins=bins,alpha=0.5,log=True,histtype='step',lw=0.5,color='k',label='Before ampl & smearing')
            val1,_,l1 = ax1.hist(imaADU[im==1],bins=bins,alpha=0.5,log=True,histtype='step',lw=0.5,color='k')
            val2,_,l2 = ax1.hist(imaADU[im==2],bins=bins,alpha=0.5,log=True,histtype='step',lw=0.5,color='k')


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
        if plot_:
            val0,_,l0 = ax1.hist(imaADU[im==0],bins=bins,alpha=0.5,label='0',log=True)
            val1,_,l1 = ax1.hist(imaADU[im==1],bins=bins,alpha=0.5,label='1',log=True)
            val2,_,l2 = ax1.hist(imaADU[im==2],bins=bins,alpha=0.5,label='2',log=True)
            ax1.hist(imaADU.flatten(),bins=bins,label='Total histogram',log=True,histtype='step',lw=1,color='k')
        else:
            val0,_ = np.histogram(imaADU[im==0],bins=bins)#,alpha=0.5,label='0',log=True)
            val1,_ = np.histogram(imaADU[im==1],bins=bins)#,alpha=0.5,label='1',log=True)
            val2,_ = np.histogram(imaADU[im==2],bins=bins)#,alpha=0.5,label='2',log=True)

        b = (bins[:-1]+bins[1:])/2
        rn_noise = (RN/(Emgain * ConversionGain)) * np.array([np.sum(val0[b>bi]) for bi in b])/np.sum(val0) #/(Emgain*ConversionGain)#/(Emgain*ConversionGain)
        # rn_noise = (RN/1)* np.array([np.sum(val0[b>bi]) for bi in b])/np.sum(val0) #/(Emgain*ConversionGain)#/(Emgain*ConversionGain)
        signal12 = flux * np.array([np.sum(val1[b>bi])+np.sum(val2[b>bi]) for bi in b])/(np.sum(val1)+np.sum(val2))
        signal1 = flux * np.array([np.sum(val1[b>bi]) for bi in b])/np.sum(val1)
        pc = np.ones(len(b))# 
        pc = ([np.sum(val1[b>bi])for bi in b]/(np.array([np.sum(val1[b>bi])for bi in b])+np.array([np.sum(val0[b>bi]) for bi in b])))
        # pc = ([np.sum(val0[b>bi])for bi in b]/(np.array([np.sum(val1[b>bi])for bi in b])+np.array([np.sum(val0[b>bi]) for bi in b])))
        true_positives =  ([np.sum(val1[b>bi])for bi in b]/(np.array([np.sum(val1[b>bi])for bi in b])+np.array([np.sum(val0[b>bi]) for bi in b])))
        fake_negatives =  ([np.sum(val1[b<bi])for bi in b]/(np.array([np.sum(val1[b<bi])for bi in b])+np.array([np.sum(val0[b<bi]) for bi in b])))
        if dark_cic_sky_noise is None:
            noise = CIC_noise**2+dark_noise**2+Sky_noise**2
        else:
            noise = dark_cic_sky_noise
        # print('noises = ',noise)
        # plt.semilogy(b,signal1)
        
        # plt.semilogy(b,signal1)
        # plt.semilogy(b,rn_noise)
        # plt.semilogy(b,signal1/np.sqrt(signal1+noise+(1*rn_noise**2)))
        # plt.semilogy(b,signal1/np.sqrt((signal1+noise+rn_noise**2)))
        # plt.semilogy(b,signal1/(signal1+0.1*noise+rn_noise**2))
        # plt.plot(b,signal1/np.sqrt(signal1+noise+np.array(rn_noise)**2))
        SNR1 = pc*signal1/np.sqrt(signal1+noise+np.array(rn_noise)**2)
        SNR12 = pc*signal12/ np.sqrt(signal12+noise+np.array(rn_noise)**2)
        SNR_analogic = flux/np.sqrt(2*flux+2*noise+(RN/(Emgain * ConversionGain))**2)
        # print('SNR_analogic = ',SNR_analogic)
        
        threshold = b[np.nanargmax(SNR1)]
        index = np.nanargmax(SNR1)
        fraction_signal = np.sum(val1[index:])/np.sum(val1)
        fraction_rn = np.sum(val0[index:])/np.sum(val0)
        true_positive, fake_negative = true_positives[index], fake_negatives[index]
        lw=3
        if plot_:
            ax2.plot(b,signal1/flux,label='Signal(Signal>T)',lw=lw)
            ax2.plot(b,np.array([np.sum(val0[b>bi]) for bi in b])/np.sum(val0),label='RN(RN>T)',lw=lw)
            # ax2.plot(b,np.array(rn_noise)**2,label='(RN(RN>T)/EMGAIN)**2',lw=lw)
            ax2.plot(b,pc,label='Fraction(T) of true positive',lw=lw)
            #ax2.plot(b,SNR1/pc,label='SNR without fraction')



            ax2.plot(b,SNR1,label='SNR1, frac(N0)=%i%%, frac(N1)=%i%%'%(100*np.sum(val0[np.nanargmax(SNR1):])/np.sum(val0),100*np.sum(val1[np.nanargmax(SNR1):])/np.sum(val1)),lw=lw)
            # ax2.plot(b,SNR12,':',label='SNR12, [N1+N2]/[N0] = %0.2f, frac(N1+N2)=%i%%'%((val1[np.nanargmax(SNR12)]+val2[np.nanargmax(SNR12)])/val0[np.nanargmax(SNR12)],100*np.sum(val1[np.nanargmax(SNR12):]+val2[np.nanargmax(SNR12):])/(np.sum(val1)+np.sum(val2))),lw=lw)
            # ax2.plot(b,SNR1/SNR_analogic,label='SNR1 PC / SNR analogic',lw=lw)
            # ax2.plot(b,SNR12/SNR_analogic,':',label='SNR12 PC / SNR analogic',lw=lw)
            # ax2.set_yscale('log')
            ax2.set_ylim(ymin=1e-5)
            
            # ax2.plot(b,SNR1,label='[N1]/[N0] = %0.2f, frac(N1)=%i%%'%(val1[np.nanargmax(SNR1)]/val0[np.nanargmax(SNR1)],100*np.sum(val1[np.nanargmax(SNR1):])/np.sum(val1)))
            # ax2.plot(b,SNR12,label='[N1+N2]/[N0] = %0.2f, frac(N1+N2)=%i%%'%((val1[np.nanargmax(SNR12)]+val2[np.nanargmax(SNR12)])/val0[np.nanargmax(SNR12)],100*np.sum(val1[np.nanargmax(SNR12):]+val2[np.nanargmax(SNR12):])/(np.sum(val1)+np.sum(val2))))

            L = ax1.legend(fontsize=7)
            ax2.legend(fontsize=7)
            ax2.set_xlabel('ADU')
            ax1.set_ylabel('#')
            ax2.set_ylabel('SNR')
            threshold = b[np.nanargmax(SNR1)]
            L.get_texts()[1].set_text('0 e- : %i%%, faction kept: %0.2f%%'%(100*values[0]/(size[0]*size[1]),100*np.sum(val0[np.nanargmax(SNR1):])/np.sum(val0)))
            L.get_texts()[2].set_text('1 e- : %i%%, faction kept: %0.2f%%'%(100*values[1]/(size[0]*size[1]),100*np.sum(val1[np.nanargmax(SNR1):])/np.sum(val1)))
            L.get_texts()[3].set_text('2 e- : %i%%, faction kept: %0.2f%%'%(100*values[2]/(size[0]*size[1]),100*np.sum(val2[np.nanargmax(SNR1):])/np.sum(val2)))
            ax1.plot([threshold,threshold],[0,np.max(val0)],':',c='k')
            ax2.plot([threshold,threshold],[0,np.nanmax(SNR1)],':',c='k')
            ax1.set_title(title+'Gain = %i, RN = %i, flux = %0.2f, Smearing=%0.1f, Threshold = %i = %0.2f$\sigma$'%(Emgain,RN,flux,self.smearing, threshold,threshold/(RN*ConversionGain)))
            ax1.set_xlim(xmin=bins.min(),xmax=7000)#bins.max())
            fig.tight_layout()

        #print(i)
        #print('flux, threshold,fractions = ',flux,threshold, fraction_signal, fraction_rn)
        print('RN, Sky_noise, Emgain, CIC_noise: ', RN, Sky_noise, Emgain, CIC_noise)
        # print("INTERP: Emgain = %0.2f, RN = %0.2f, flux = %0.2f, smearing = %0.2f"%(Emgain,RN,flux, self.smearing))

        # print("MEASURE: Threshold = %0.2f, Signal fraction = %0.2f, RN fraction = %0.2f, snr_ratio = %0.2f"%(threshold/(RN*ConversionGain), fraction_signal, fraction_rn, np.nanmax(SNR1/SNR_analogic)))

        return threshold/(RN*ConversionGain), fraction_signal, fraction_rn, np.nanmax((SNR1/pc)/SNR_analogic), true_positive, fake_negative
 


    def interpolate_optimal_threshold(self,flux = 0.1,dark_cic_sky_noise=None,plot_=False,title='',i=0):
        #self.Signal_el if np.isscalar(self.Signal_el) else 0.3
        Emgain = self.EM_gain #if np.isscalar(self.EM_gain) else self.EM_gain[i]
        RN= self.RN #if np.isscalar(self.RN) else self.RN[i]#80
        CIC_noise = self.CIC_noise #if np.isscalar(self.CIC_noise) else self.CIC_noise[i]
        dark_noise = self.Dark_current_noise #if np.isscalar(self.Dark_current_noise) else self.Dark_current_noise[i]
         
        try:
            Sky_noise = self.Sky_noise_pre_thresholding #if np.isscalar(self.Sky_noise_pre_thresholding) else self.Sky_noise_pre_thresholding[i]
        except AttributeError:
            raise AttributeError('You must use counting_mode=True to use compute_optimal_threshold method.')

        noise_value = CIC_noise**2+dark_noise**2+Sky_noise**2
        n=6
        gains=np.linspace(800,2500,n)
        rons=np.linspace(30,120,n)
        fluxes=np.linspace(0.01,0.7,n)
        smearings=np.linspace(0,2,n)
        noise=np.linspace(0.002,0.05,n)
        if n==6:
            coords = (gains, rons, fluxes, smearings)
            point = (Emgain, RN, flux, self.smearing)            
        elif n==5:
            coords = (gains, rons, fluxes, smearings,noise)
            point = (Emgain, RN, flux, self.smearing,noise_value)
            
        if ~np.isscalar(noise_value) |  ~np.isscalar(self.smearing) | ~np.isscalar(Emgain) | ~np.isscalar(RN):
            point = np.repeat(np.zeros((4,1)), 50, axis=1).T
            point[:,0] =  self.EM_gain
            point[:,1] = self.RN
            point[:,2] = flux
            point[:,3] = self.smearing
        # np.repeat(point[:,  np.newaxis], 2, axis=1).T
        # print(len(point),len(coords))
# table_threshold = Table.read("threshold_6.fits")
# table_snr = Table.read("snr_max_6.fits")
# table_fraction_rn = Table.read("fraction_rn_6.fits")
# table_fraction_flux = Table.read("fraction_flux_6.fits")
        fraction_rn =interpn(coords, table_fraction_rn, point,bounds_error=False,fill_value=None)
        fraction_signal =interpn(coords, table_fraction_flux, point,bounds_error=False,fill_value=None)
        threshold = interpn(coords, table_threshold, point,bounds_error=False,fill_value=None)
        snr_ratio = interpn(coords, table_snr, point,bounds_error=False,fill_value=None)
        #print(i)
        #print('flux, threshold,fractions = ',flux,threshold, fraction_signal, fraction_rn)
        #print('RN, Sky_noise, Emgain, CIC_noise: ', RN, Sky_noise, Emgain, CIC_noise)
        # print(threshold,fraction_signal,fraction_rn,snr_ratio)
        # print(snr_ratio)
        # print("INTERP: Emgain = %0.2f, RN = %0.2f, flux = %0.2f, smearing = %0.2f"%(np.unique(Emgain),np.unique(RN),np.unique(flux),np.unique( self.smearing)))
        # print("INTERP: Threshold = %0.2f, Signal fraction = %0.2f, RN fraction = %0.2f, snr_ratio = %0.2f"%(np.unique(threshold),np.unique(fraction_signal),np.unique(fraction_rn),np.unique(snr_ratio)))
        return threshold, fraction_signal, fraction_rn, snr_ratio#np.nanmax(SNR1/SNR_analogic)
 
   

np.seterr(invalid='ignore')
class ExposureTimeCalulator(widgets.HBox):
     
    def __init__(self, follow_temp=True, exposure_time=50, acquisition_time=2, Sky_LU=4, Signal=5.57e-18, EM_gain=1400,RN=109,CIC_charge=0.005, Dard_current=0.08,readout_time=1.5,x_axis='exposure_time',counting_mode=False,smearing=0.7,Additional_background_2018=0,temperature=-100):
        super().__init__()
        self.output = widgets.Output()
        self.Additional_background_2018=Additional_background_2018
        time=np.linspace(1,150)
        i = np.argmin(abs(time - exposure_time))
        self.follow_temp=follow_temp
        # temperature =  -100#np.linspace(-110,-80)
        # Dard_current = 10**np.poly1d([0.07127906, 6.83562573])(np.linspace(-110,-80))
        # smearing = np.poly1d([-0.0306087, -2.2226087])(np.linspace(-110,-80))

        Nominal = Observation(exposure_time=time,counting_mode=counting_mode, Signal=Signal, EM_gain=EM_gain, RN=RN, CIC_charge=CIC_charge, Dard_current=Dard_current, Sky_LU=10**Sky_LU, readout_time=readout_time, acquisition_time = acquisition_time,Additional_background_2018=self.Additional_background_2018,smearing=smearing,i=i,temperature=temperature)#np.linspace(-110,-80))#
        #flight_background_damping = 0.9, Additional_background_2018 = 0.0007
        self.x = time#Nominal.exposure_time#np.linspace(0, 2 * np.pi, 100)
        self.fig = Nominal.PlotNoise(x=x_axis)
        args, _, _, locals_ = inspect.getargvalues(inspect.currentframe())
        self.v=[]
        # print(locals_[x_axis])
        for i, ax in enumerate(self.fig.axes):
            if i==2:
                try:
                    label = 'SNR max = %0.1f\nT=%0.1f sigma\nSignal kept=%i%%, RN kept=%i%%'%(Nominal.SNR.max(),Nominal.n_threshold[i], 100*Nominal.Photon_fraction_kept[i], 100*Nominal.RN_fraction_kept[i])
                except TypeError:
                    label = 'SNR max = %0.1f\nT=%0.1f sigma\nSignal kept=%i%%, RN kept=%i%%'%(Nominal.SNR.max(),Nominal.n_threshold, 100*Nominal.Photon_fraction_kept, 100*Nominal.RN_fraction_kept)
                
                self.v.append(ax.axvline(locals_[x_axis],ls=':',c='k',label=label))
            else:
                self.v.append(ax.axvline(locals_[x_axis],ls=':',c='k'))
        ax.legend(loc='upper right')
        self.ax0 =  self.fig.axes[0]
        self.ax1 =  self.fig.axes[1]
        self.ax2 =  self.fig.axes[2]
    
        self.fig.canvas.toolbar_position = 'bottom'
        style={}#{'description_width': 'initial'} 
        width = '400px'
        small = '247px'
        self.exposure_time = widgets.IntSlider( min=1, max=150,value=exposure_time, layout=Layout(width=width),description='Texp (s)')#'$t_{exp}$ (s)')
        self.acquisition_time = widgets.FloatSlider( min=0.1, max=10,value=acquisition_time, layout=Layout(width=width),description='Taq (h)')#'$t_{aq}$ (h)')
        self.Sky_LU = widgets.FloatLogSlider( min=4, max=7,value=Sky_LU,base=10, style =style, layout=Layout(width=width),description='Sky (LU)')
#         self.Signal = widgets. FloatLogSlider( min=-18, max=-15,value=1.25e-17, base=10,step=1e-18, style = style, layout=Layout(width='500px'),description='Flux')
        self.Signal = widgets.Dropdown(options=[('Bright Galex star (mU~15)', 2.22e-14), ('Extremely Bright QSO (mU~15)', 5.08e-15), ('Bright QSO (mU~19.5)', 3.51e-16),('Bright galaxy (mU~22.5)', 2.22e-17), ('Regular galaxy (mU~24)', 5.57e-18), ('Low SB galaxy (mU~25)', 2.2e-18), ('Cosmic web (1e-21)', 1e-21)],value=Signal,description='Flux')
        self.EM_gain = widgets.IntSlider( min=200, max=2000,value=EM_gain, style = style, layout=Layout(width=width),description='EM gain')
        self.RN = widgets.IntSlider( min=30, max=120,value=RN, style = style, layout=Layout(width=width),description='Read noise')
        self.CIC_charge = widgets.FloatSlider( min=0.003, max=0.05,value=CIC_charge,style = style, layout=Layout(width=width),description='CIC charge',step=0.001,readout_format='.3f')#$_{charge}$
        self.Dard_current = widgets.FloatSlider( min=0.01, max=17,value=Dard_current, style = style, layout=Layout(width=width),description='Dard current',step=0.0011,readout_format='.2f')#$_{current}$
        self.readout_time = widgets.FloatSlider( min=1.5, max=20,value=readout_time, style = style, layout=Layout(width=width),description=r'RO time (s)',step=0.1)#$_{time}$
        options = ['exposure_time','Sky_LU','acquisition_time',"Signal","EM_gain","RN","CIC_charge","Dard_current" ,"readout_time","smearing","temperature"] if follow_temp else ['exposure_time','Sky_LU','acquisition_time',"Signal","EM_gain","RN","CIC_charge","Dard_current" ,"readout_time","smearing"]
        self.x_axis=widgets.Dropdown(options=options,value=x_axis,description='X axis', layout=Layout(width=small))
#         self.mode=widgets.Dropdown(options=['Flight 2018','Nominal 2022 10MHz','Nominal 2022 100MHz'],value='Nominal 2022 10MHz',description='X axis')
        self.counting_mode = widgets.Checkbox(value=counting_mode,description='PC Threshold',disabled=False, layout=Layout(width=small))
        # smearing = widgets.FloatSlider( min=0, max=1.5,value=0.7, layout=Layout(width='500px'),description='CTE not yet',step=0.001)#'$t_{aq}$ (h)') #widgets.Dropdown(options=[0.2,0.7,1.2],value=0.7,description='Smearing (pix)')

        self.smearing = widgets.FloatSlider( min=0, max=2,value=smearing, layout=Layout(width=width),description='Smearing',step=0.01)#'$t_{aq}$ (h)') #widgets.Dropdown(options=[0.2,0.7,1.2],value=0.7,description='Smearing (pix)')      
        self.temperature = widgets.FloatSlider( min=-120, max=-80,value=-90, style = style,description=r'Temp (C)',step=0.1, layout=Layout(width=width))
        if ~self.counting_mode.value:
            self.smearing.layout.visibility = 'hidden'
    
        wids = widgets.interactive(self.update,x_axis=self.x_axis,smearing=self.smearing,counting_mode=self.counting_mode,exposure_time=self.exposure_time,Sky_LU=self.Sky_LU,acquisition_time=self.acquisition_time,Signal=self.Signal,EM_gain=self.EM_gain,RN=self.RN, CIC_charge=self.CIC_charge, Dard_current=self.Dard_current, readout_time=self.readout_time,temperature=self.temperature)
        if follow_temp:
            self.Dard_current.value = 10**np.poly1d([0.07127906, 6.83562573])(self.temperature.value)
            # self.smearing.value = np.poly1d([-0.0306087, -2.2226087])(self.temperature.value)
            self.smearing.value = np.poly1d([-0.0453913, -3.5573913])(self.temperature.value)
            controls = VBox([HBox([self.x_axis,self.Signal,self.counting_mode,self.temperature]),   HBox([self.Sky_LU,self.EM_gain,self.smearing])     ,  HBox([self.acquisition_time,self.Dard_current,self.CIC_charge]),    HBox([self.exposure_time,self.RN,self.readout_time]) ] )
        else:
            controls = VBox([HBox([self.x_axis,self.Signal,self.counting_mode]),   HBox([self.Sky_LU,self.EM_gain,self.smearing])     ,  HBox([self.acquisition_time,self.Dard_current,self.CIC_charge]),    HBox([self.exposure_time,self.RN,self.readout_time]) ] )
            

        
        # wids = widgets.interact(self.update,x_axis=x_axis,counting_mode=False,exposure_time=exposure_time,Sky_LU=Sky_LU,acquisition_time=acquisition_time,Signal=Signal,EM_gain=EM_gain,RN=RN, CIC_charge=CIC_charge, Dard_current=Dard_current, readout_time=readout_time)
        display(HBox([self.output,controls]))


        


    def update(self, x_axis,counting_mode,Sky_LU,acquisition_time,Signal,EM_gain,RN,CIC_charge,Dard_current,readout_time,exposure_time,smearing,temperature):
        """Draw line in plot"""
        with self.output:
            self.smearing.layout.visibility = 'visible' if counting_mode else 'hidden'
            if self.follow_temp:
                self.Dard_current.value = 10**np.poly1d([0.07127906, 6.83562573])(temperature)
                # self.smearing.value = np.poly1d([-0.0306087, -2.2226087])(temperature)
                self.smearing.value = np.poly1d([-0.0453913, -3.5573913])(temperature)

            args, _, _, locals_ = inspect.getargvalues(inspect.currentframe())
   
            value = locals_[x_axis]

            if x_axis == 'temperature':
                temperature=np.linspace(self.temperature.min, self.temperature.max)
                Dard_current = 10**np.poly1d([0.07127906, 6.83562573])(temperature)
                # smearing = np.poly1d([-0.0306087, -2.2226087])(temperature)
                smearing = np.poly1d([-0.0453913, -3.5573913])(temperature)
            # else:
            #     temperature = self.temperature.value
            #     self.Dard_current.value = 10**np.poly1d([0.07127906, 6.83562573])(temperature)
            #     self.smearing.value = np.poly1d([-0.0306087, -2.2226087])(temperature)
            #     Dard_current = 10**np.poly1d([0.07127906, 6.83562573])(temperature)
            #     smearing = np.poly1d([-0.0306087, -2.2226087])(temperature)

            # if counting_mode:
            #     # self.x_axis.options = ['exposure_time','Sky_LU','acquisition_time',"Signal","EM_gain","RN","CIC_charge","Dard_current" ,"readout_time","smearing"]
            # else:
            #     # if self.x_axis.value == 'smearing':
            #     #     self.x_axis.options = ['exposure_time','Sky_LU','acquisition_time',"Signal","EM_gain","RN","CIC_charge","Dard_current" ,"readout_time"]
            #     #     self.x_axis.value = "exposure_time"
            #     self.smearing.layout.visibility = 'hidden'
            #     x_axis = self.x_axis.value


#             self.fig.suptitle('pompo,')
            # self.ax0.set_title('Gain=%i, Texp=%i, CIC=%0.3f, Dark=%0.3f, RN=%i, Signal=%0.2E, Sky=%0.2E, e-/pix=%0.2E'%(EM_gain,exposure_time,CIC_charge,Dard_current,RN,Signal,Sky_LU,self.el_per_pix),y=0.97)
            title = 'Gain'#=%i, Texp=%i, CIC=%0.3f, Dark=%0.3f, RN=%i, Signal=%0.2E, Sky=%0.2E'%(EM_gain,exposure_time,CIC_charge,Dard_current,RN,Signal,Sky_LU)
            if x_axis == 'exposure_time':
                exposure_time=np.linspace(self.exposure_time.min,self.exposure_time.max)
            if x_axis == 'Sky_LU':
                Sky_LU=np.logspace(3,7)
            if x_axis == 'Signal':
                Signal=np.logspace(-18,-15)
            if x_axis == 'EM_gain':
                EM_gain=np.linspace(self.EM_gain.min,self.EM_gain.max)
            if x_axis == 'acquisition_time':
                acquisition_time=np.linspace(0.1,6)
            if x_axis == 'RN':
                RN=np.linspace(self.RN.min,self.RN.max)
            elif x_axis == 'CIC_charge':
                CIC_charge=np.linspace(self.CIC_charge.min,self.CIC_charge.max)
            if x_axis == 'Dard_current':
                Dard_current=np.linspace(self.Dard_current.min,self.Dard_current.max)
            if x_axis == 'readout_time':
                readout_time=np.linspace(self.readout_time.min,self.readout_time.max)
            if x_axis == 'smearing':
                smearing=np.linspace(self.smearing.min,self.smearing.max)

            if (x_axis == 'Sky_LU') | (x_axis == 'Signal') | (x_axis == 'CIC_charge') | (x_axis == 'Dard_current'):
                self.ax0.set_xscale('log')
#                 self.ax2.set_yscale('log')
            else:
                self.ax0.set_xscale('linear')
#                 self.ax2.set_yscale('linear')
            args, _, _, locals_ = inspect.getargvalues(inspect.currentframe())
            new_value = locals_[x_axis]
            arg = np.argmin(abs(new_value - value))
            # print('argi = ',value,arg)
            new = Observation(exposure_time=exposure_time,Sky_LU=Sky_LU, acquisition_time=acquisition_time,counting_mode=counting_mode,Signal=Signal,EM_gain=EM_gain,RN=RN,CIC_charge=CIC_charge,Dard_current=Dard_current,readout_time=readout_time,smearing=smearing,Additional_background_2018=self.Additional_background_2018,i=arg,temperature=temperature)
            self.ax0.set_title(title + ", e-/pix=%0.2E"%(new.el_per_pix[arg]),y=0.97)
            for v in self.v:
                v.set_xdata([value,value])
            arg = np.argmin(abs(getattr(new,x_axis) - value))
            try:
                label = 'SNRmax=%0.2f, SNR=%0.2f\nT=%0.1f sigma\nSignal kept=%i%%, RN kept=%i%%\nThresholding gain=%0.2f'%(new.SNR.max(),new.SNR[arg],new.n_threshold, 100*new.Photon_fraction_kept, 100*new.RN_fraction_kept, new.gain_thresholding)
            except TypeError:
                label = 'SNRmax=%0.2f, SNR=%0.2f\nT=%0.1f sigma\nSignal kept=%i%%, RN kept=%i%%\nThresholding gain=%0.2f'%(new.SNR.max(),new.SNR[arg],new.n_threshold[arg], 100*new.Photon_fraction_kept[arg], 100*new.RN_fraction_kept[arg], new.gain_thresholding[arg])
                
            self.v[-1].set_label(label)
            max_,min_=[],[]
            self.ax0.set_title(-1)

            # for i,name in enumerate(["signal_noise","Dark_current_noise", "Additional_background_noise","Sky_noise", "CIC_noise", "RN_final"]):
            for i,name in enumerate(["Signal","Dark current","Sky", "CIC", "Read noise"]): #, "Add. background"
                self.ax0.lines[i].set_xdata(locals_[x_axis])
                self.ax0.lines[i].set_ydata(new.noises[:,i])
                max_.append(np.nanmax(new.noises[:,i]))
                min_.append(np.nanmin(new.noises[:,i]))
                # try:
                # except AttributeError:
                #     self.fig.axes[0].lines[i].set_xdata(-100)

                # self.fig.axes[0].lines[i].set_ydata(getattr(new,name)*new.factor)
                            # ax1.plot(getattr(self,x), self.noises[:,i],label='%s: %i (%0.1f%%)'%(name,self.noises[self.i,i],self.percents[i,self.i]),lw=lw,alpha=0.8,c=c)

                # self.ax0.lines[i].set_label('%s: %i (%0.1f%%)'%(name,new.noises[new.i,i],new.percents[i,new.i]))
            self.ax0.legend(loc='upper right')
                # print(name)
            # import sys
            # sys.exit()
                #### self.fig.axes[0].lines[i].set_label('%s: %i (%0.1f%%)'%(name,(getattr(new,name)*new.factor)[arg],new.percents[i,arg]))

            self.ax0.set_ylim((-0.5,np.max(max_)))
            self.ax2.legend(loc='upper right')
            self.ax2.collections.clear()#cla()#
            self.ax1.collections.clear()#cla()#
            # self.ax1.stackplot(temperature,new.snrs * np.array(new.noises).T[:-1,:]**2/new.Total_noise_final**2,alpha=0.7,colors=colors)
            # self.ax2.stackplot(temperature,new.snrs * np.array(new.noises).T[:-1,:]**2/new.Total_noise_final**2,alpha=0.7,colors=colors)
            self.ax2.stackplot(locals_[x_axis],new.snrs * np.array(new.noises).T[:-1,:]**2/new.Total_noise_final**2,alpha=0.7,colors=colors)
            labels = ['%s: %0.3f (%0.1f%%)'%(name,getattr(new,"electrons_per_pix")[new.i,j],100*getattr(new,"electrons_per_pix")[new.i,j]/np.sum(getattr(new,'electrons_per_pix')[new.i,:])) for j,name in enumerate(["Signal","Dark current","Sky", "CIC", "Read noise"])]#, "Add. background"
            self.ax1.stackplot(locals_[x_axis],  np.array(new.electrons_per_pix).T,alpha=0.7,colors=colors,labels=labels)
            # print(locals_[x_axis],(new.snrs * np.array(new.noises).T[:-1,:]**2/new.Total_noise_final**2).shape)
            # self.x=locals_[x_axis]
            # self.etc =  np.array(new.electrons_per_pix).T
            self.ax1.legend(loc='upper right')
            self.ax1.set_ylim((0,np.max(np.sum(getattr(new,'electrons_per_pix'),axis=1))))
            self.ax2.set_xlabel(x_axis)
            self.ax2.set_xlim((np.min(locals_[x_axis]),np.max(locals_[x_axis])))
            # self.ax2.set_xlim((0,1))
            self.ax2.set_ylim((0,np.max(getattr(new,'SNR'))))


# ETC = ExposureTimeCalulator(EM_gain=1700,RN=65, smearing=0.8,Dard_current=1,x_axis='exposure_time',counting_mode=False,follow_temp=True)
# %load_ext line_profiler
self =  Observation(EM_gain=1700,RN=65, smearing=0.6,Dard_current=1,counting_mode=True)
# %lprun -f Observation.compute_optimal_threshold 
self.compute_optimal_threshold(plot_=True)
# %lprun -f  self.compute_optimal_threshold(plot_=True)
# print("\n")
# a=Observation(EM_gain=1700,RN=65, smearing=0.8,Dard_current=1,counting_mode=True).interpolate_optimal_threshold(plot_=True)


#%%

n=10

from astropy.io import fits
from tqdm import tqdm
# n=6
gains=np.linspace(800,2500,n)
rons=np.linspace(30,120,n)
fluxes=np.linspace(0.01,0.7,n)
smearings=np.linspace(0,2,n)

print(gains,rons,fluxes,smearings)
#normally I should also make evolve 

threshold2 = np.zeros((len(gains),len(rons),len(fluxes),len(smearings)))
fraction_flux2 = np.zeros((len(gains),len(rons),len(fluxes),len(smearings)))
fraction_gain2 = np.zeros((len(gains),len(rons),len(fluxes),len(smearings)))
snrs_2 = np.zeros((len(gains),len(rons),len(fluxes),len(smearings)))
true_pos = np.zeros((len(gains),len(rons),len(fluxes),len(smearings)))
fake_negs = np.zeros((len(gains),len(rons),len(fluxes),len(smearings)))

gi = np.zeros((len(gains),len(rons),len(fluxes),len(smearings)))
ri = np.zeros((len(gains),len(rons),len(fluxes),len(smearings)))
fi = np.zeros((len(gains),len(rons),len(fluxes),len(smearings)))
si = np.zeros((len(gains),len(rons),len(fluxes),len(smearings)))

for i in tqdm(range(len(gains))):
    for j  in tqdm(range(len(rons))):
        for k in range(len(fluxes)):
            for l in range(len(smearings)):
                t,f1,f2,snr,tp,fn = Observation(exposure_time=np.array([50,50]), EM_gain=gains[i], RN=rons[j],smearing=smearings[l],counting_mode=True).compute_optimal_threshold(flux=fluxes[k],plot_=False)#1.5)#2022
                # t,f1,f2,snr = Observation(exposurenhu_time=50, EM_gain=gains[i], RN=rons[j],smearing=smearings[l],counting_mode=True).interpolate_optimal_threshold(flux=fluxes[k],plot_=False)#1.5)#2022
                # t,f1,f2,snr = np.unique(t), np.unique(f1), np.unique(f2), np.unique(snr)
                threshold2[i,j,k,l] = t
                fraction_flux2[i,j,k,l] = f1
                fraction_gain2[i,j,k,l] = f2    
                snrs_2[i,j,k,l] = snr  
                true_pos[i,j,k,l]  =tp
                fake_negs[i,j,k,l] = fn

                gi[i,j,k,l] = gains[i]
                ri[i,j,k,l] = rons[j]
                fi[i,j,k,l] = fluxes[k]
                si[i,j,k,l] = smearings[l]
                
# fits.HDUList([fits.PrimaryHDU(threshold2)]).writeto('interp_threshold_%i.fits'%(n),overwrite=True)
# fits.HDUList([fits.PrimaryHDU(fraction_flux2)]).writeto('interp_fraction_flux_%i.fits'%(n),overwrite=True)
# fits.HDUList([fits.PrimaryHDU(fraction_gain2)]).writeto('interp_fraction_rn_%i.fits'%(n),overwrite=True)
# fits.HDUList([fits.PrimaryHDU(snrs_2)]).writeto('interp_snr_max_%i.fits'%(n),overwrite=True)
fits.HDUList([fits.PrimaryHDU(threshold2)]).writeto('threshold_%i.fits'%(n),overwrite=True)
fits.HDUList([fits.PrimaryHDU(fraction_flux2)]).writeto('fraction_flux_%i.fits'%(n),overwrite=True)
fits.HDUList([fits.PrimaryHDU(fraction_gain2)]).writeto('fraction_rn_%i.fits'%(n),overwrite=True)
fits.HDUList([fits.PrimaryHDU(snrs_2)]).writeto('snr_max_%i.fits'%(n),overwrite=True)
fits.HDUList([fits.PrimaryHDU(true_pos)]).writeto('true_pos_%i.fits'%(n),overwrite=True)
fits.HDUList([fits.PrimaryHDU(fake_negs)]).writeto('fake_negs_%i.fits'%(n),overwrite=True)
