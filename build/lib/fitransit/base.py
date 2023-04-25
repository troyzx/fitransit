from fitransit.singlefit import SingleFit

import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from astroquery.mast import Observations
import rebound
import matplotlib.colors as colors
import scipy.stats
from multiprocessing import get_context
from uncertainties import ufloat
from pytransit.lpf.tesslpf import TESSLPF
from pytransit.orbits import epoch
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

planeturl = "https://exo.mast.stsci.edu/api/v0.1/exoplanets/"
dvurl = "https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/"
url = planeturl + "/identifiers/"
header = {}

mj_to_ms = 9.5e-4
me_to_ms = 3.0e-6
rj_to_rs = 0.102792236
rs_to_AU = 0.00464913034
daytos = 24 * 60 * 60


def get_id(planet_name: str):
    myparams = {"name": planet_name}
    url = planeturl + "/identifiers/"
    r = requests.get(url=url, params=myparams, headers=header)
    # print(r.headers.get('content-type'))
    planet_names = r.json()
    ticid = planet_names["tessID"]

    return ticid


def get_prop(planet_name: str, tic: int):
    url = planeturl + planet_name + "/properties/"
    r = requests.get(url=url, headers=header)
    url = planeturl + planet_name + "/properties/"
    r = requests.get(url=url, headers=header)

    return r.json()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


# create a new directory and save data, while also checking if the folder
# exists and asking the user if they want to create a new folder, and if the
# file already exists, asking if the user wants to overwrite it
def save_df_data(dir_path, file_name, df_data):

    # Check if the directory exists:
    if not os.path.exists(dir_path):
        # The directory doesn't exist, so ask the user if they want to create
        # it:
        create_dir = input(
            "The directory doesn't exist. Would you like to create it? (y/n) ")
        if create_dir.lower() == "y":
            os.makedirs(dir_path)
        else:
            print("Unable to save data.")
            exit()

    # Check if the file already exists:
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        # The file already exists, so ask the user if they want to overwrite
        # it:
        overwrite_file = input(
            "The file already exists. Do you want to overwrite it? (y/n) ")
        if overwrite_file.lower() != "y":
            print("Unable to save data.")
            exit()

    # Save some example data to the file:
    df_data.to_csv(file_path, index=False)

    print("Data saved successfully!")


def read_data(name: str):
    with open(name, "r") as file:
        return file


def getn(unfloat):
    return unfloat.n


def gets(unfloat):
    return unfloat.s


getn_v = np.vectorize(getn)
gets_v = np.vectorize(gets)
epoch_v = np.vectorize(epoch)

mj_to_ms = 9.5e-4
me_to_ms = 3.0e-6
rj_to_rs = 0.102792236
rs_to_AU = 0.00464913034
daytos = 24 * 60 * 60


class fitlpf:
    def __init__(self, planet_name: str, datadir=None):
        if datadir is None:
            datadir = "./data/" + planet_name
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
        self.prop = get_prop(planet_name, ticid)
        transit_time = self.prop[0]["transit_time"] + 2.4e6 + 0.5
        transit_time_err = max(
            self.prop[0]["transit_time_lower"],
            self.prop[0]["transit_time_upper"]
        )
        orbital_period = self.prop[0]["orbital_period"]
        orbital_period_err = max(
            self.prop[0]["orbital_period_lower"],
            self.prop[0]["orbital_period_upper"]
        )
        self.period = ufloat(orbital_period, orbital_period_err)
        self.zero_epoch = ufloat(transit_time, transit_time_err)
        self.ticid = ticid
        self.print_parameters()

    def print_parameters(self):
        planet_prop = self.prop
        print(self.planet_name + " Properties")
        print(
            "Stellar Mass \t\t%f \t\t%s"
            % (planet_prop[0]["Ms"], planet_prop[0]["Ms_unit"])
        )
        print(
            "Planet Mass \t\t%f \t\t%s"
            % (planet_prop[0]["Mp"], planet_prop[0]["Mp_unit"])
        )
        print(
            "Planet Orbital Period \t%f \t\t%s"
            % (planet_prop[0]["orbital_period"],
               planet_prop[0]["orbital_period_unit"])
        )
        print(
            "Transit Time \t\t%f \t\t%s"
            % (
                planet_prop[0]["transit_time"] + 0.5,
                planet_prop[0]["transit_time_unit"],
            )
        )
        print("Planet Mass Reference: %s" % (planet_prop[0]["Mp_ref"]))

    def download_data(self):
        observations = Observations.query_object(
            self.planet_name, radius="0 deg")
        obs_wanted = (observations["dataproduct_type"] == "timeseries") \
            & (
            observations["obs_collection"] == "TESS"
        )
        print(observations[obs_wanted]["obs_collection", "project", "obs_id"])
        data_products = Observations.get_product_list(observations[obs_wanted])
        products_wanted = Observations.filter_products(
            data_products, productSubGroupDescription=["DVT", "LC"]
        )

        print(products_wanted["productFilename"])
        manifest = Observations.download_products(
            products_wanted, download_dir=self.datadir
        )
        return manifest
        print("\nfinished!")

    def de(self, niter=200, npop=30, datadir=None):
        zero_epoch = self.zero_epoch
        period = self.period

        self.lpf = TESSLPF(
            self.planet_name,
            datadir,
            tic=self.ticid,
            zero_epoch=zero_epoch.n,
            period=period.n,
            use_pdc=True,
            nsamples=2,
            bldur=0.25,
        )

        ep = epoch(self.lpf.times[0].mean(), self.zero_epoch.n, self.period.n)
        tc = zero_epoch + ep * period

        self.lpf.set_prior(
            "tc", "NP", tc.n, 0.005
        )  # Wide normal prior on the transit center
        self.lpf.set_prior(
            "p", "NP", period.n, period.s
        )  # Wide normal prior on the orbital period
        self.lpf.set_prior("rho", "UP", 0, 1)
        # Uniform prior on the stellar density
        self.lpf.set_prior("k2", "UP", 0.0, 0.2**2)
        # Uniform prior on the area ratio
        self.lpf.set_prior(
            "gp_ln_in", "UP", -2, 1
        )  # Uniform prior on the GP input scale
        self.lpf.optimize_global(niter=niter, npop=npop)

    def plot_original_data(self):
        timea = self.lpf.timea
        fluxa = self.lpf.ofluxa
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        plt.plot(timea, fluxa)
        return fig

    def fit_single(self, i, niter=100, npop=50, mcmc_repeats=4):
        single = SingleFit(
            self.planet_name + str(i) + "th",
            None,
            self.lpf.times[i],
            self.lpf.fluxes[i],
        )
        ep = epoch(single.timea.mean(), self.zero_epoch.n, self.period.n)
        tc = self.zero_epoch + ep * self.period

        single.set_prior(
            "tc", "NP", tc.n, tc.s
        )  # Wide normal prior on the transit center
        single.set_prior(
            "p", "NP", self.period.n, self.period.s
        )  # Wide normal prior on the orbital period
        single.set_prior("rho", "UP", 0, 1)
        # Uniform prior on the stellar density
        single.set_prior("k2", "UP", 0.0, 0.2**2)
        # Uniform prior on the area ratio
        single.optimize_global(niter=niter, npop=npop)
        single.sample_mcmc(2500, thin=25, repeats=mcmc_repeats)

        return single

    def fit_singles(self, niter=100, npop=50):
        self.fit_single_v = np.vectorize(self.fit_single)
        self.singles = self.fit_single_v(np.arange(len(self.lpf.times)))
        self.post_samples = []
        self.tcs = []
        for single in self.singles:
            df = single.posterior_samples()
            self.post_samples.append(df)
            tc = ufloat(df["tc"].mean(), df["tc"].std())
            self.tcs.append(tc)

        self.epochs = \
            epoch_v(getn_v(self.tcs), self.zero_epoch.n, self.period.n)

    def get_posterior_samples(self):
        self.post_samples = []
        self.tcs = []
        for single in self.singles:
            df = single.posterior_samples()
            self.post_samples.append(df)
            tc = ufloat(df["tc"].mean(), df["tc"].std())
            self.tcs.append(tc)
        self.epochs = epoch_v(
            getn_v(self.tcs), self.zero_epoch.n, self.period.n)

    def calculate_ttv(self):
        day_to_s = 24 * 60 * 60
        tcs = getn_v(self.tcs)
        epochs = self.epochs
        self.ttv_err = (
            gets_v(self.tcs) + self.period.s * (epochs - epochs[0])
        ) * day_to_s

        def f_1(x, A, B):
            return A * x + B

        A, B = curve_fit(f_1, epochs - epochs[0], tcs - tcs[0])[0]

        self.ttv_mcmc_raw = (tcs - tcs[0] - A * (epochs - epochs[0]) + B) \
            * day_to_s
        self.ttv_mcmc = self.ttv_mcmc_raw - self.ttv_mcmc_raw.mean()

    def plot_tcs(self, plot_zero_epoch=False):
        epochs = self.epochs
        tcs = self.tcs
        if plot_zero_epoch is True:
            tcs = np.insert(tcs, 0, self.zero_epoch)
            epochs = np.insert(epochs, 0, 0)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        # plt.plot(epochs,getn_v(self.tcs),'.')
        plt.errorbar(epochs, getn_v(tcs), yerr=gets_v(tcs), fmt="o")
        plt.xlabel("Epoch")
        plt.ylabel(r"$t_c$ (MJD)")

    def plot_ttv_re(
        self, plot_zero_epoch=False, set_epoch_zero=False, remove_baseline=True
    ):
        epochs = self.epochs
        ttv_mcmc = self.ttv_mcmc
        ttv_raw = self.ttv_mcmc_raw
        ttv_err = self.ttv_err

        if plot_zero_epoch is True:
            ttv_mcmc = np.insert(ttv_mcmc, 0, 0)
            ttv_mcmc_raw = np.insert(ttv_raw, 0, 0)
            ttv_err = np.insert(ttv_err, 0, self.zero_epoch.s)
            epochs = np.insert(epochs, 0, 0)

        if set_epoch_zero is True:
            if plot_zero_epoch is False:
                epochs = epochs - epochs[0]
            else:
                raise Exception(
                    "Cannot set plot_zero_epoch and set_epoch_zero \
                        = True at the same time"
                )

        if remove_baseline is True:
            ttv = ttv_mcmc
        else:
            ttv = ttv_mcmc_raw

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        plt.errorbar(epochs, ttv, yerr=ttv_err, fmt="o")

        # plt.plot(epochs,getn_v(re)*day_to_s,linestyle='',marker='.',markersize=12)
        plt.xlabel("Epoch")
        plt.ylabel("residual (s)")
