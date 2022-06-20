# get prop
import numpy as np
import requests
import matplotlib.pyplot as plt
from IPython.display import display, HTML

#get data
from astroquery.mast import Observations
from astroquery.mast import Catalogs
from astropy.io import fits
from astropy import table
from copy import deepcopy
from corner import corner

planeturl = "https://exo.mast.stsci.edu/api/v0.1/exoplanets/"
dvurl = "https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/"
url = planeturl + "/identifiers/"
header={}

# mcmc
import pandas as pd
import seaborn as sb
from pathlib import Path
from uncertainties import ufloat
from scipy.stats import norm
from pytransit.lpf.tesslpf import TESSLPF, BaseLPF, fold, downsample_time
from pytransit.orbits import epoch
from scipy.optimize import curve_fit
from . import singlefit



def get_id(planet_name: str):
    
    myparams = {"name":planet_name}
    url = planeturl + "/identifiers/"
    r = requests.get(url = url, params = myparams, headers = header)
    # print(r.headers.get('content-type'))
    planet_names = r.json()
    ticid = planet_names['tessID']
    
    return ticid


def get_prop(planet_name: str, tic: int):
    
    url = planeturl + planet_name + "/properties/"
    r = requests.get(url = url, headers = header)
    url = planeturl + planet_name + "/properties/"
    r = requests.get(url = url, headers = header)
    
    return r.json()

def getn(unfloat):
    return unfloat.n

def gets(unfloat):
    return unfloat.s


getn_v = np.vectorize(getn)
gets_v = np.vectorize(gets)
epoch_v = np.vectorize(epoch)



class fitlpf:
    
    
    
    def __init__(self, planet_name: str, datadir = None):
        
        if datadir == None:
            datadir = './data/' + planet_name.replace(' ','')
        self.planet_name = planet_name
        self.ticid = None
        self.period = None
        self.zero_epoch = None
        self.prop = []
        self.datadir = datadir
        self.lpf = None
        self.singles = []
        self.post_samples = []
        self.tcs = []
        self.epochs = []
            
    
    
    def get_parameter(self):
         
        planet_name = self.planet_name
        ticid = get_id(planet_name)
        self.prop  = get_prop(planet_name, ticid)
        transit_time = self.prop[0]['transit_time'] + 2.4e6 + 0.5
        transit_time_err = max(self.prop[0]['transit_time_lower'],self.prop[0]['transit_time_upper'])
        orbital_period = self.prop[0]['orbital_period']
        orbital_period_err = max(self.prop[0]['orbital_period_lower'],self.prop[0]['orbital_period_upper'])
        self.period = ufloat(orbital_period, orbital_period_err)
        self.zero_epoch = ufloat(transit_time, transit_time_err)
        self.ticid = ticid
        self.print_parameters()
    
    
    
    def print_parameters(self):

        planet_prop = self.prop
        print(self.planet_name + " Properties")
        print("Stellar Mass \t\t%f \t\t%s" % \
              (planet_prop[0]['Ms'], planet_prop[0]['Ms_unit'] ) )
        print("Planet Mass \t\t%f \t\t%s" % \
              (planet_prop[0]['Mp'], planet_prop[0]['Mp_unit'] ) )
        print("Planet Orbital Period \t%f \t\t%s" % \
              (planet_prop[0]['orbital_period'], \
               planet_prop[0]['orbital_period_unit'] ) )
        print("Transit Time \t\t%f \t\t%s" % \
              (planet_prop[0]['transit_time'] + 0.5, \
               planet_prop[0]['transit_time_unit'] ) )
        print("Planet Mass Reference: %s" % \
              (planet_prop[0]['Mp_ref']))

    
    
    def download_data(self):
        
        observations = Observations.query_object(self.planet_name,radius = "0 deg")
        obs_wanted = (observations['dataproduct_type'] \
                      == 'timeseries') & (observations['obs_collection'] == 'TESS')
        print( observations[obs_wanted]['obs_collection', 'project','obs_id'] )
        data_products = Observations.get_product_list(observations[obs_wanted])
        products_wanted = Observations.filter_products(data_products, 
                                            productSubGroupDescription=["DVT","LC"])

        print(products_wanted["productFilename"])
        manifest = Observations.download_products(products_wanted,download_dir=self.datadir)
        print('\nfinished!')

    
    
    def de(self, niter=200, npop=30, datadir=None):
        
        zero_epoch = self.zero_epoch
        period = self.period

        self.lpf = TESSLPF(self.planet_name,datadir,tic=self.ticid,zero_epoch=zero_epoch.n,period=period.n,use_pdc=True,nsamples=2, bldur=0.25)
        
        ep = epoch(self.lpf.times[0].mean(), self.zero_epoch.n, self.period.n)
        tc = zero_epoch + ep*period

        self.lpf.set_prior('tc', 'NP', tc.n,     0.005)          # Wide normal prior on the transit center
        self.lpf.set_prior('p',  'NP', period.n, period.s)  # Wide normal prior on the orbital period
        self.lpf.set_prior('rho', 'UP', 0, 1)               # Uniform prior on the stellar density
        self.lpf.set_prior('k2', 'UP', 0.0, 0.2**2)             # Uniform prior on the area ratio
        self.lpf.set_prior('gp_ln_in', 'UP', -2, 1)             # Uniform prior on the GP input scale
        self.lpf.optimize_global(niter=niter, npop=npop)

    
    
    def plot_original_data(self):
        
        timea = self.lpf.timea
        fluxa = self.lpf.ofluxa
        
        fig,ax = plt.subplots(figsize=(10,6),dpi=200)
        plt.plot(timea, fluxa)

        return fig

    
    
    def fit_single(self, i, niter=100, npop=50, mcmc_repeats=4):
        
        single = singlefit.SingleFit(self.planet_name + str(i) + 'th', None, self.lpf.times[i], self.lpf.fluxes[i])
        ep = epoch(single.timea.mean(), self.zero_epoch.n, self.period.n)
        tc = self.zero_epoch + ep*self.period

        single.set_prior('tc', 'NP', tc.n,     tc.s)          # Wide normal prior on the transit center
        single.set_prior('p',  'NP', self.period.n, self.period.s)  # Wide normal prior on the orbital period
        single.set_prior('rho', 'UP', 0, 1)               # Uniform prior on the stellar density
        single.set_prior('k2', 'UP', 0.0, 0.2**2)             # Uniform prior on the area ratio
        single.optimize_global(niter=niter, npop=npop)
        single.sample_mcmc(2500, thin=25, repeats=mcmc_repeats)

        return single

    
    
    def fit_singles(self,niter=100, npop=50):

        self.fit_single_v = np.vectorize(self.fit_single)
        self.singles = self.fit_single_v(np.arange(len(self.lpf.times)))
        self.post_samples = []
        self.tcs = []
        for single in self.singles:
            df = single.posterior_samples()
            self.post_samples.append(df)
            tc = ufloat(df['tc'].mean(),df['tc'].std())
            self.tcs.append(tc)
        
        self.epochs = epoch_v(getn_v(self.tcs),self.zero_epoch.n, self.period.n)
    
    
    
    def get_posterior_samples(self):
        
        self.post_samples = []
        self.tcs = []
        for single in self.singles:
            df = single.posterior_samples()
            self.post_samples.append(df)
            tc = ufloat(df['tc'].mean(),df['tc'].std())
            self.tcs.append(tc)
        self.epochs = epoch_v(getn_v(self.tcs),self.zero_epoch.n, self.period.n)

    
    
    def plot_ttv(self,plot_zero_epoch=False):
        
        epochs = self.epochs
        tcs = self.tcs
        if plot_zero_epoch == True:
            tcs = np.insert(tcs,0,self.zero_epoch)
            epochs = np.insert(epochs,0,0)
        fig,ax = plt.subplots(figsize=(10,6),dpi=200)
        # plt.plot(epochs,getn_v(self.tcs),'.')
        plt.errorbar(epochs,getn_v(tcs),yerr=gets_v(tcs),fmt='o')
        plt.xlabel('Epoch')
        plt.ylabel(r'$t_c$ (MJD)')

        return fig
    
    
    
    def plot_ttv_re(self,plot_zero_epoch=False,set_epoch_zero=False):
        
        epochs = self.epochs
        tcs = self.tcs
        
        if plot_zero_epoch == True:
            tcs = np.insert(tcs,0,self.zero_epoch)
            epochs = np.insert(epochs,0,0)
        
        re = getn_v(tcs) - epochs * self.period - self.zero_epoch
        
        if set_epoch_zero == True:
            if plot_zero_epoch == False:
                epochs = epochs - epochs[0]
            else:
                raise Exception('Cannot set plot_zero_epoch and set_epoch_zero = True at the same time')
        
        daytos = 24*60*60
        fig,ax = plt.subplots(figsize=(10,6),dpi=200)
        plt.errorbar(epochs,getn_v(re)*daytos,yerr=gets_v(re)*daytos,fmt='o')

        # plt.plot(epochs,getn_v(re)*daytos,linestyle='',marker='.',markersize=12)
        plt.xlabel('Epoch')
        plt.ylabel('residual (s)')


        

