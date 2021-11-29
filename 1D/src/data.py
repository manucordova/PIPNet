###########################################################################
###                               PIPNet                                ###
###                      Data generation functions                      ###
###                        Author: Manuel Cordova                       ###
###                       Last edited: 2021-10-12                       ###
###########################################################################

import numpy as np
import time

import torch

class PIPDataset(torch.utils.data.Dataset):
    """
    Generating PIP dataset
    """

    def __init__(self, **kwargs):
        """
        Initialize the class
        """

        # Set parameters
        self.__dict__.update(kwargs)

        # Extract time domain points
        self.t = np.arange(0, self.td, 1) / self.Fs

        # Get frequency domain points
        self.df = self.Fs / self.td
        self.f = np.arange(0, self.td, 1) * self.df

        return

    def gen_iso_peak(self):
        """
        Generate an isotropic peak as a sum of Gaussians
        """

        # Randomly get the number of Gaussians in a peak (uniform distribution)
        dn = self.pmax - self.pmin + 1
        n = self.pmin + int(np.random.rand() * dn)

        # Get isotropic shift
        dw0 = self.shift_range[1] - self.shift_range[0]
        w0 = self.shift_range[0] + (np.random.rand() * dw0)

        # Get shift displacement for each Gaussian (normal distribution)
        if self.ds > 0.:
            ws = w0 + (np.random.randn(n) * self.ds)
        else:
            ws = np.ones(n) * w0

        if isinstance(self.lw[0], list):

            if self.iso_p_peakwise:
                # Randomly select broadening range
                p = np.random.random()
                i = 0
                p_tot = self.iso_p[i]
                while p > p_tot:
                    i += 1
                    p_tot += self.iso_p[i]

                # Get broadening with the same range for each Gaussian
                dlw = self.lw[i][1] - self.lw[i][0]
                lws = self.lw[i][0] + (np.random.rand(n) * dlw)

            else:
                lws = []

                for _ in range(n):
                    # Randomly select broadening range
                    p = np.random.random()
                    i = 0
                    p_tot = self.iso_p[i]
                    while p > p_tot:
                        i += 1
                        p_tot += self.iso_p[i]

                    # Get broadening with the same range for each Gaussian
                    dlw = self.lw[i][1] - self.lw[i][0]
                    lws.append(self.lw[i][0] + (np.random.rand() * dlw))
                lws = np.array(lws)



        else:
            # Get broadening for each Gaussian (uniform distribution)
            dlw = self.lw[1]-self.lw[0]
            lws = self.lw[0] + (np.random.rand(n) * dlw)

        # Get phase for each Gaussian (normal distribution)
        if self.phase > 0.:
            ps = np.random.randn(n) * self.phase
        else:
            ps = np.zeros(n)

        if self.debug:
            print(f"  Generating a peak of {n} Gaussians around {w0} Hz...")
            print("           v[Hz] | dv [Hz] | phi [rad]")
            for i, (w, lw, p) in enumerate(zip(ws, lws, ps)):
                print(f"    G {i+1: 2.0f} | {w:.3f} | {lw:.5f} | {p:.7f}")

        # Build FID
        fid = np.zeros(len(self.t), dtype=complex)
        fs = []

        for w, lw, p in zip(ws, lws, ps):
            # Frequency
            f = np.exp(w * self.t * 2 * np.pi * 1j)
            # Broadening
            f *= np.exp(-1 * (self.t * lw) ** 2)
            # Phase
            f *= np.exp(-1j * p)
            f[0] /= 2
            fs.append(f)
            fid += f

        fid /= n

        return fid, fs

    def gen_iso_spectra(self):
        """
        Generate isotropic spectra
        """

        # Randomly get the number of peaks (uniform distribution)
        dn = self.nmax - self.nmin + 1
        n = int(np.random.rand() * dn) + self.nmin

        if self.debug:
            print(f"Generating a spectrum made of {n} peaks...")

        # Generate peaks
        fids = np.empty((n, len(self.t)), dtype=complex)
        for i in range(n):
            fids[i], _ = self.gen_iso_peak()

        # Fourier transform
        isos = np.real(np.fft.fft(fids, axis=1))

        # Randomize intensity
        da = self.iso_int[1] - self.iso_int[0]
        a = self.iso_int[0] + (np.random.rand(n) * da)
        isos *= a[:, np.newaxis]

        # Make isotropic spectra positive
        if self.positive:
            isos[np.real(isos) < 0.] = 0.

        iso = np.expand_dims(np.sum(isos, axis=0), 0)

        return n, isos, iso

    def gen_mas_params(self, n):
        """
        Generate MAS parameters (rate, broadening, shift)

        Input:      - n     Number of peaks in the spectrum

        Outputs:    - wr    MAS rates
                    - ls    MAS-dependent Lorentzian broadening for each peak
                    - gs    MAS-dependent Gaussian broadening for each peak
                    - ss    MAS-dependent shift for each peak
                    - ps    MAS-dependent phase for each spectrum/peak
        """

        # Generate MAS rates (linear or uniformly random)
        if self.random_mas:
            dw = self.mas_w_range[1] - self.mas_w_range[0]
            w0 = self.mas_w_range[0]
            wr = w0 + (np.sort(np.random.rand(self.nw)) * dw)
        else:
            wr = np.linspace(self.mas_w_range[0], self.mas_w_range[1], self.nw)

        if self.peakwise_phase:
            # Generate phase of MAS spectra (normal distribution)
            ps = np.random.randn(self.nw, n) * self.mas_phase
        else:
            # Generate phase of MAS spectra (normal distribution)
            ps = np.random.randn(self.nw) * self.mas_phase

        if isinstance(self.mas_l_range[0], list):
            # Randomly select broadening range
            ls = []
            gs = []
            ss = []
            for _ in range(n):
                # Randomly select broadening range
                p = np.random.random()
                i = 0
                p_tot = self.mas_p[i]
                while p > p_tot:
                    i += 1
                    p_tot += self.mas_p[i]

                # Generate MAS-dependent Lorentzian broadening (uniform distribution)
                dl = self.mas_l_range[i][1] - self.mas_l_range[i][0]
                l0 = self.mas_l_range[i][0]
                ls.append(l0 + (np.random.rand() * dl))

                # Generate MAS-dependent Gaussian broadening (uniform distribution)
                dg = self.mas_g_range[i][1] - self.mas_g_range[i][0]
                g0 = self.mas_g_range[i][0]
                gs.append(g0 + (np.random.rand() * dg))

                # Generate MAS-dependent shift
                ds = self.mas_s_range[i][1] - self.mas_s_range[i][0]
                s0 = self.mas_s_range[i][0]
                ss.append(s0 + (np.random.rand() * ds))

            ls = np.array(ls)
            gs = np.array(gs)
            ss = np.array(ss)

        else:
            # Generate MAS-dependent Lorentzian broadening (uniform distribution)
            dl = self.mas_l_range[1] - self.mas_l_range[0]
            l0 = self.mas_l_range[0]
            ls = l0 + (np.random.rand(n) * dl)

            # Generate MAS-dependent Gaussian broadening (uniform distribution)
            dg = self.mas_g_range[1] - self.mas_g_range[0]
            g0 = self.mas_g_range[0]
            gs = g0 + (np.random.rand(n) * dg)

            # Generate MAS-dependent shift
            ds = self.mas_s_range[1] - self.mas_s_range[0]
            s0 = self.mas_s_range[0]
            ss = s0 + (np.random.rand(n) * ds)

        if self.debug:
            print("\n  Generated MAS rates: " + ", ".join([f"{w:6.0f}" for w in wr]) + " Hz")
            if self.peakwise_phase:
                print("  Generated MAS phases:")
                for i, p in enumerate(ps.T):
                    print(f"    Peak {i+1: 3.0f}           " + ", ".join([f"{pi: 2.3f}" for pi in p]) + " rad")
            else:
                print("  Generated MAS phases:" + ", ".join([f"{p: 2.3f}" for p in ps]) + " rad")

            print("\n  Generating MAS-dependent parameters...")
            print("                 l [Hz^2] |  g [Hz^3] |  s [Hz^2]")
            for i, (l, g, s) in enumerate(zip(ls, gs, ss)):
                print(f"    Peak {i+1: 3.0f}    {l: 2.2e} | {g: 2.2e} | {s: 2.2e}")

        return wr, ls, gs, ss, ps

    def mas_broaden(self, specs, wr, ls, gs, ss, ps):
        """
        Broaden isotropic spectra with MAS-dependent parameters

        Inputs: - specs         Isotropic spectrum for each peak
                - wr            MAS rates
                - ls            MAS-dependent Lorentzian broadening for each peak
                - gs            MAS-dependent Gaussian broadening for each peak
                - ss            MAS-dependent shift for each peak
                - ps            MAS-dependent phase for each spectrum/peak

        Output: - brd_specs     Broadened spectra
        """

        n_mas = wr.shape[0]
        n_pks, n_pts = specs.shape
        r = np.arange(1., n_pts+1) / n_pts

        brd_specs = np.empty((n_mas, n_pks, n_pts), dtype=complex)

        # Get MAS-dependent matrices of parameters
        L = np.tile(ls, n_pts).reshape(n_pts, n_pks).T
        G = np.tile(gs, n_pts).reshape(n_pts, n_pks).T
        S = np.tile(ss, n_pts).reshape(n_pts, n_pks).T

        for i, p in zip(range(n_mas), ps):

            # Shift and Lorentzian broadening
            e1 = (1j * 2 * np.pi * (r * self.Fs + (S / wr[i]))) - L / wr[i]
            e1 = np.einsum("ij,k->ijk", e1, self.t)

            # Gaussian broadening
            e2 = np.einsum("ij,k->ijk", -1 * G / wr[i], np.square(self.t)).astype(complex)

            # Apply exponential
            brd_specs[i] = np.matmul(np.expand_dims(specs, 1), np.exp(e1 + e2)).squeeze()

            # Apply phase
            if self.peakwise_phase:
                P = np.tile(p, n_pts).reshape(n_pts, n_pks).T
            else:
                P = np.ones((n_pks, n_pts)) * p

            brd_specs[i] *= np.exp(-1j * P)

        brd_specs[:, :, 0] /= 2.

        # Make output (n_max, n_dim, n_pts)
        data = np.fft.fft(brd_specs, axis=-1)
        if self.encode_imag:
            output = np.empty((n_mas, 2, n_pts))
            output[:, 0, :] = np.mean(np.real(data), axis=1)
            output[:, 1, :] = np.mean(np.imag(data), axis=1)

        else:
            output = np.empty((n_mas, 1, n_pts))
            output[:, 0, :] = np.mean(np.real(data), axis=1)

        return output

    def normalize_spectra(self, iso, specs, brd_specs):

        # Get real spectra integrals
        int0 = np.sum(iso)
        ints = np.sum(brd_specs[:,0], axis=1)

        # Normalize isotropic spectrum
        iso /= self.iso_norm
        iso += self.offset
        specs /= self.iso_norm
        specs += self.offset

        # Normalize broadened spectra
        fac = (int0 / ints)[:, np.newaxis, np.newaxis]
        brd_specs *= fac / self.brd_norm
        brd_specs += self.offset

        return iso, specs, brd_specs

    def finalize_spectra(self, iso, specs, brd_specs, wr):

        # Add noise
        if self.noise > 0.:
            brd_specs += np.random.normal(loc=0., scale=self.noise, size=brd_specs.shape)

        # Smooth spectra edges
        if self.smooth_end_len > 0:
            for i in range(self.smooth_end_len):
                brd_specs[:, :, i] *= i / self.smooth_end_len
                brd_specs[:, :, -(i+1)] *= i / self.smooth_end_len

        # Encode MAS rate
        if self.encode_w:
            wr_enc = np.repeat(np.expand_dims(wr, (1, 2)), brd_specs.shape[-1], axis=2)

            # Encode inverse MAS rate
            if self.wr_inv:
                wr_enc = self.wr_factor / wr_enc
            # Normalize MAS rate
            elif self.norm_wr:
                wr_enc -= self.mas_w_range[0]
                wr_enc /= self.mas_w_range[1] - self.mas_w_range[0]

            # Append encoded MAS rate to spectra
            brd_specs = np.append(brd_specs, wr_enc, axis=1)

        # Add dummy single-peak isotropic spectra to get the same array shape for all data generation
        if specs.shape[0] < self.nmax:
            specs = np.pad(specs, ((0, self.nmax - specs.shape[0]), (0, 0)))

        return (torch.from_numpy(brd_specs.astype(np.float32)),
                torch.from_numpy(specs.astype(np.float32)),
                torch.from_numpy(iso.astype(np.float32)))

    def __len__(self):
        """
        Dummy function for pytorch to work properly
        """

        return int(1e12)

    def __getitem__(self, _):
        """
        Generate an input
        """

        # Generate isotropic spectrum
        n, specs, iso = self.gen_iso_spectra()

        # Generate MAS-dependent parameters
        wr, ls, gs, ss, ps = self.gen_mas_params(n)

        # Broaden isotropic spectrum with MAS-dependent parameters
        brd_specs = self.mas_broaden(specs, wr, ls, gs, ss, ps)

        # Set the minimum of each spectrum to zero
        brd_specs -= np.min(brd_specs[:, 0], axis=1)[:, np.newaxis, np.newaxis]

        # Normalize spectra
        iso, specs, brd_specs = self.normalize_spectra(iso, specs, brd_specs)

        return self.finalize_spectra(iso, specs, brd_specs, wr)
