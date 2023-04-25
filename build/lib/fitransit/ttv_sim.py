import rebound
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import get_context
from tqdm.auto import tqdm

mj_to_ms = 9.5e-4
me_to_ms = 3.0e-6
rj_to_rs = 0.102792236
rs_to_AU = 0.00464913034
daytos = 24 * 60 * 60


"""
This code defines a class, ttv_analy, which is built to perform simulations of
exoplanetary systems using the REBOUND N-body integrator. The class has
several methods for running simulations, calculating transit times, and
generating plots of the results.

The main functionality of the class is to analyze the stability of planetary
systems by evaluating the Mean Exponential Growth of Nearby Orbits (MEGNO)
values. This indicator can help determine the chaotic behavior of a system
and find potentially stable configurations.
"""


def get_chi2(ttv_rebound, epoch, ttv_mcmc, ttv_err):
    # print(ttv_rebound)
    rangea = range(epoch[-1] - epoch[0])
    T = [ttv_rebound[np.array(epoch - epoch[0]) + a] for a in rangea]
    chi2 = (((T - ttv_mcmc) ** 2) / ttv_err**2).sum(axis=1)
    return chi2.min()


def get_rms(ttv_rebound):
    rms = np.sqrt(np.mean(ttv_rebound**2))
    return rms


get_chi2_v = np.vectorize(
    get_chi2, excluded=["epoch", "ttv_mcmc", "ttv_err"], signature="(n)->()"
)
get_rms_v = np.vectorize(get_rms, signature="(n)->()")


class ttv_sim:
    def __init__(self, epochs, ttv_mcmc, ttv_err, rs, mp2s, prop, N=80):
        self.epochs = epochs
        self.ttv_mcmc = ttv_mcmc
        self.ttv_err = ttv_err
        self.rs = rs
        self.mp2s = mp2s
        self.N = N
        self.crit = scipy.stats.chi2.ppf(0.997, len(ttv_mcmc))
        self.prop = prop
        self.a1 = prop[0]["orbital_distance"]
        self.t1 = prop[0]["orbital_period"]
        self.mp1 = prop[0]["Mp"]
        self.ms = prop[0]["Ms"]
        self.megno_dt = 1 / 20
        self.megno_runtime = 1e4
        self.ttv_rebound = []

    def calculate_rebound(self, par):
        r, mp2 = par
        sim = rebound.Simulation()
        ms = self.prop[0]["Ms"]
        mp1 = self.prop[0]["Mp"]
        a1 = self.prop[0]["orbital_distance"]
        a2 = a1 * r ** (2 / 3)
        rstar = self.prop[0]["Rs"]
        rp = self.prop[0]["Rp"]

        sim = rebound.Simulation()
        sim.integrator = "whfast"
        sim.ri_whfast.safe_mode = 0

        # Collision
        sim.collision = "direct"
        sim.add(m=ms, r=rstar * rs_to_AU)  # Star
        sim.add(
            m=mp1 * mj_to_ms, r=rp * rj_to_rs * rs_to_AU, a=a1, e=0
        )  # Primary planet
        sim.add(m=mp2 * me_to_ms, a=a2, e=0)  # Companion planet
        sim.move_to_com()
        sim.exit_max_distance = 5.0

        period_min = min([sim.particles[1].P, sim.particles[2].P])
        N = (self.epochs[-1] - self.epochs[0]) * 2
        transittimes = np.zeros(N)
        p = sim.particles
        i = 0
        while i < N:
            y_old = p[1].y - p[0].y
            t_old = sim.t
            try:
                sim.integrate(sim.t + period_min / 4)
                # check for transits every <period_min / 4>
                # Note that <period_min / 4> is shorter than one inner planet's orbit
            except rebound.Escape:
                # print("Escape at r={}, mp2={} when i={}".format(r,mp2,i))
                break
            except rebound.Collision:
                # print("Collide at r={}, mp2={} when i={}".format(r,mp2,i))
                break
            t_new = sim.t
            if y_old * (p[1].y - p[0].y) < 0.0 and p[1].x - p[0].x > 0.0:
                # sign changed (y_old*y<0), planet in front of star (x>0)
                while t_new - t_old > 1e-9:
                    # bisect until prec of 1e-9 reached
                    if y_old * (p[1].y - p[0].y) < 0.0:
                        t_new = sim.t
                    else:
                        t_old = sim.t
                    try:
                        sim.integrate((t_new + t_old) / 2.0)
                    except:
                        break
                transittimes[i] = sim.t
                i += 1
                try:
                    sim.integrate(sim.t + 5e-5)
                except:
                    break
        c, m = np.linalg.lstsq(
            np.vstack([np.ones(N), range(N)]).T, transittimes, rcond=None
        )[0]
        ttv_rebound = (transittimes - m * np.array(range(N)) - c) * (
            3600 * 24.0 * 365.0 / 2.0 / np.pi
        )
        return ttv_rebound

    def get_ttv_rebound_all(self, number_of_thread):
        parameters = []
        for mp2 in self.mp2s:
            for r in self.rs:
                parameters.append((r, mp2))
        with get_context("fork").Pool(number_of_thread) as p:
            self.ttv_results = list(
                tqdm(p.imap(self.calculate_rebound, parameters), total=len(parameters))
            )
        self.ttv_rebound = np.array(self.ttv_results)
        return self.ttv_rebound

    def get_m_crit(self):
        epoch = self.epochs
        ttv_mcmc = self.ttv_mcmc
        ttv_err = self.ttv_err
        rs = self.rs
        mp2s = self.mp2s
        ttv_results = self.ttv_results

        chi2 = get_chi2_v(
            ttv_rebound=np.array(self.ttv_results),
            epoch=epoch,
            ttv_mcmc=ttv_mcmc,
            ttv_err=ttv_err,
        )
        chi2_crit = scipy.stats.chi2.ppf(0.997, len(ttv_mcmc))

        rms = get_rms_v(ttv_results)
        rms_crit = np.sqrt(np.mean(ttv_mcmc**2))

        chi2[rms == 0] = None
        rms[rms == 0] = None

        chi2_2d = np.array(chi2).reshape(len(mp2s), len(rs))
        rms_2d = np.array(rms).reshape(len(mp2s), len(rs))

        m_crit_chi2 = []
        for r in rs:
            r_idx = np.where(rs == r)
            for mp2 in mp2s:
                mp2_idx = np.where(mp2s == mp2)
                if chi2_2d[mp2_idx, r_idx] < chi2_crit:
                    pass
                else:
                    m_crit_chi2.append(mp2)
                    break

        m_crit_rms = []
        for r in rs:
            r_idx = np.where(rs == r)
            for mp2 in mp2s:
                mp2_idx = np.where(mp2s == mp2)
                if rms_2d[mp2_idx, r_idx] < rms_crit:
                    pass
                else:
                    m_crit_rms.append(mp2)
                    break

        self.m_crit_chi2 = np.array(m_crit_chi2)
        self.m_crit_rms = np.array(m_crit_rms)

        return self.m_crit_chi2, self.m_crit_rms

    def simulation_m(self, par):
        r, mp2 = par  # unpack parameters
        prop = self.prop
        ms = prop[0]["Ms"]
        mp1 = prop[0]["Mp"]
        a1 = prop[0]["Ms"]
        a2 = a1 * r ** (2 / 3)

        sim = rebound.Simulation()
        sim.integrator = "whfast"
        sim.ri_whfast.safe_mode = 0
        sim.add(m=ms)  # Star
        sim.add(m=mp1 * mj_to_ms, a=a1, e=0)  # Primary planet
        sim.add(m=mp2 * me_to_ms, a=a2, e=0)  # Companion planet
        sim.move_to_com()

        period_min = min([sim.particles[1].P, sim.particles[2].P])
        sim.dt = self.megno_dt * period_min

        sim.init_megno()
        sim.exit_max_distance = 20.0
        try:
            sim.integrate(self.megno_runtime * period_min, exact_finish_time=0)
            # integrate for <runtime>, integrating to the nearest
            # timestep for each output to keep the timestep constant
            # and preserve WHFast's symplectic nature
            megno = sim.calculate_megno()
            return megno
        except rebound.Escape:
            return 10.0
        # At least one particle got ejected,
        # returning large MEGNO.

    # Run the MEGNO simulations for all parameter combinations
    def run_megno(self, number_of_threads):
        rs = self.rs
        mp2s = self.mp2s
        parameters = []
        for mp2 in mp2s:
            for r in rs:
                parameters.append((r, mp2))

        with get_context("fork").Pool(number_of_threads) as p:
            self.megno_results = list(
                tqdm(p.imap(self.simulation_m, parameters), total=len(parameters))
            )
        return self.megno_results

    # Plot the MEGNO results as a 2D color map
    def plot_megno(self):
        rs = self.rs
        mp2s = self.mp2s
        results2d = np.array(self.megno_results).reshape(len(rs), len(mp2s))
        fig, ax = plt.subplots(figsize=(7, 5))
        extent = [min(rs), max(rs), min(mp2s), max(mp2s)]
        ax.set_xlim(extent[0], extent[1])
        ax.set_xlabel("$P_2/P_1$")
        ax.set_ylim(extent[2], extent[3])
        ax.set_ylabel("Mass [$M_j$]")
        im = ax.imshow(
            results2d,
            interpolation="none",
            vmin=1.9,
            vmax=10,
            cmap="RdYlGn_r",
            origin="lower",
            aspect="auto",
            extent=extent,
            alpha=0.8,
        )
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("MEGNO $\\langle Y \\rangle$")
        plt.grid()
        new_ticks = [1.5, 2, 2.5, 3, 3.3, 3.5, 3.8, 4]
        plt.xticks(new_ticks)
        plt.xlabel(r"$P_2/P_1$")
        plt.ylabel(r"$M_2$ [$M_j$]")