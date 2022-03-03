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

        self.mas_w2 = False

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

    def gen_mas2_params(self, n):
        """
        Generate second-order MAS parameters (rate, broadening, shift)

        Input:      - n     Number of peaks in the spectrum

        Outputs:    - ls    MAS^2-dependent Lorentzian broadening for each peak
                    - gs    MAS^2-dependent Gaussian broadening for each peak
                    - ss    MAS^2-dependent shift for each peak
        """

        if isinstance(self.mas2_l_range[0], list):
            # Randomly select broadening range
            ls2 = []
            gs2 = []
            ss2 = []
            for _ in range(n):
                # Randomly select broadening range
                p = np.random.random()
                i = 0
                p_tot = self.mas2_p[i]
                while p > p_tot:
                    i += 1
                    p_tot += self.mas2_p[i]

                # Generate MAS-dependent Lorentzian broadening (uniform distribution)
                dl = self.mas2_l_range[i][1] - self.mas2_l_range[i][0]
                l0 = self.mas2_l_range[i][0]
                ls2.append(l0 + (np.random.rand() * dl))

                # Generate MAS-dependent Gaussian broadening (uniform distribution)
                dg = self.mas2_g_range[i][1] - self.mas2_g_range[i][0]
                g0 = self.mas2_g_range[i][0]
                gs2.append(g0 + (np.random.rand() * dg))

                # Generate MAS-dependent shift
                ds = self.mas2_s_range[i][1] - self.mas2_s_range[i][0]
                s0 = self.mas2_s_range[i][0]
                ss2.append(s0 + (np.random.rand() * ds))

            ls2 = np.array(ls2)
            gs2 = np.array(gs2)
            ss2 = np.array(ss2)

        else:
            # Generate MAS-dependent Lorentzian broadening (uniform distribution)
            dl = self.mas2_l_range[1] - self.mas2_l_range[0]
            l0 = self.mas2_l_range[0]
            ls2 = l0 + (np.random.rand(n) * dl)

            # Generate MAS-dependent Gaussian broadening (uniform distribution)
            dg = self.mas2_g_range[1] - self.mas2_g_range[0]
            g0 = self.mas2_g_range[0]
            gs2 = g0 + (np.random.rand(n) * dg)

            # Generate MAS-dependent shift
            ds = self.mas2_s_range[1] - self.mas2_s_range[0]
            s0 = self.mas2_s_range[0]
            ss2 = s0 + (np.random.rand(n) * ds)

        if self.debug:
            print("\n  Generating MAS^2-dependent parameters...")
            print("                 l [Hz^2] |  g [Hz^3] |  s [Hz^2]")
            for i, (l, g, s) in enumerate(zip(ls2, gs2, ss2)):
                print(f"    Peak {i+1: 3.0f}    {l: 2.2e} | {g: 2.2e} | {s: 2.2e}")

        return ls2, gs2, ss2

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

    def mas2_broaden(self, specs, wr, ls, ls2, gs, gs2, ss, ss2, ps):
        """
        Broaden isotropic spectra with MAS^2-dependent parameters

        Inputs: - specs         Isotropic spectrum for each peak
                - wr            MAS rates
                - ls            MAS-dependent Lorentzian broadening for each peak
                - ls2           MAS-dependent Lorentzian broadening for each peak
                - gs            MAS-dependent Gaussian broadening for each peak
                - gs2           MAS-dependent Gaussian broadening for each peak
                - ss            MAS-dependent shift for each peak
                - ss2           MAS-dependent shift for each peak
                - ps            MAS-dependent phase for each spectrum/peak

        Output: - brd_specs     Broadened spectra
        """

        n_mas = wr.shape[0]
        n_pks, n_pts = specs.shape
        r = np.arange(1., n_pts+1) / n_pts

        brd_specs = np.empty((n_mas, n_pks, n_pts), dtype=complex)

        # Get MAS-dependent matrices of parameters
        L = np.tile(ls, n_pts).reshape(n_pts, n_pks).T
        L2 = np.tile(ls2, n_pts).reshape(n_pts, n_pks).T
        G = np.tile(gs, n_pts).reshape(n_pts, n_pks).T
        G2 = np.tile(gs2, n_pts).reshape(n_pts, n_pks).T
        S = np.tile(ss, n_pts).reshape(n_pts, n_pks).T
        S2 = np.tile(ss2, n_pts).reshape(n_pts, n_pks).T

        for i, p in zip(range(n_mas), ps):

            # Shift and Lorentzian broadening
            e1 = (1j * 2 * np.pi * (r * self.Fs + (S / wr[i]) + (S2 / (wr[i]**2)))) - L / wr[i] - L2 / (wr[i]**2)
            e1 = np.einsum("ij,k->ijk", e1, self.t)

            # Gaussian broadening
            e2 = np.einsum("ij,k->ijk", -1 * G / wr[i] - G2 / (wr[i]**2), np.square(self.t)).astype(complex)

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

        if self.mas_w2 and np.random.random() < self.mas_w2_p:
            ls2, gs2, ss2 = self.gen_mas2_params(n)
            # Broaden isotropic spectrum with MAS-dependent parameters
            brd_specs = self.mas2_broaden(specs, wr, ls, ls2, gs, gs2, ss, ss2, ps)
        else:
            # Broaden isotropic spectrum with MAS-dependent parameters
            brd_specs = self.mas_broaden(specs, wr, ls, gs, ss, ps)

        # Set the minimum of each spectrum to zero
        brd_specs -= np.min(brd_specs[:, 0], axis=1)[:, np.newaxis, np.newaxis]

        # Normalize spectra
        iso, specs, brd_specs = self.normalize_spectra(iso, specs, brd_specs)

        return self.finalize_spectra(iso, specs, brd_specs, wr)




class PIPDatasetGLS(torch.utils.data.Dataset):
    """
    Generating PIP dataset with GLS broadening
    """

    def __init__(self, **kwargs):
        """
        Initialize the class
        """

        self.mas_w2 = False

        self.mas_lw_noise = 0.

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
        isos = np.fft.fft(fids, axis=1)

        # Randomize intensity
        da = self.iso_int[1] - self.iso_int[0]
        a = self.iso_int[0] + (np.random.rand(n) * da)
        isos *= a[:, np.newaxis]

        iso = np.expand_dims(np.sum(isos, axis=0), 0)

        return n, isos, iso

    def select_index_with_prob(self, prob):

        # Randomly select range
        p = np.random.random()
        i = 0
        p_tot = prob[i]
        while p > p_tot:
            i += 1
            p_tot += prob[i]
        return i

    def gen_mas0_params(self, n):
        """
        Generate MAS-independent parameters (broadening, shift)

        Input:      - n     Number of peaks in the spectrum

        Outputs:    - lws    MAS-independent GLS broadening for each peak
                    - ms    MAS-independent GLS mixing for each peak
                    - ss    MAS-independent shift for each peak
                    - ps    MAS-independent phase for each peak
        """

        if isinstance(self.mas0_lw_range[0], list):
            # Randomly select broadening range
            lws = []
            for _ in range(n):
                # Randomly select broadening range
                i = self.select_index_with_prob(self.mas0_lw_p)
                # Generate MAS-dependent GLS broadening (uniform distribution)
                dlw = self.mas0_lw_range[i][1] - self.mas0_lw_range[i][0]
                lw0 = self.mas0_lw_range[i][0]
                lws.append(lw0 + (np.random.rand() * dlw))
            lws = np.array(lws)

        else:
            dlw = self.mas0_lw_range[1] - self.mas0_lw_range[0]
            lw0 = self.mas0_lw_range[0]
            lws = lw0 + (np.random.rand(n) * dlw)

        if isinstance(self.mas0_m_range[0], list):
            ms = []
            for _ in range(n):
                # Randomly select GLS mixing range
                i = self.select_index_with_prob(self.mas0_m_p)
                # Generate MAS-dependent GLS mixing (uniform distribution)
                dm = self.mas0_m_range[i][1] - self.mas0_m_range[i][0]
                m0 = self.mas0_m_range[i][0]
                ms.append(m0 + (np.random.rand() * dm))
            ms = np.array(ms)

        else:
            # Generate MAS-dependent GLS mixing (uniform distribution)
            dm = self.mas0_m_range[1] - self.mas0_m_range[0]
            m0 = self.mas0_m_range[0]
            ms = m0 + (np.random.rand(n) * dm)

        if isinstance(self.mas0_s_range[0], list):
            ss = []
            for _ in range(n):
                # randomly select shift range
                i = self.select_index_with_prob(self.mas0_s_p)
                # Generate MAS-dependent shift
                ds = self.mas0_s_range[i][1] - self.mas0_s_range[i][0]
                s0 = self.mas0_s_range[i][0]
                ss.append(s0 + (np.random.rand() * ds))
            ss = np.array(ss)

        else:
            # Generate MAS-dependent shift
            ds = self.mas0_s_range[1] - self.mas0_s_range[0]
            s0 = self.mas0_s_range[0]
            ss = s0 + (np.random.rand(n) * ds)

        if self.debug:
            print("\n  Generating MAS-independent parameters...")
            print("                lw [Hz] |     m     |  s [Hz]")
            for i, (lw, m, s) in enumerate(zip(lws, ms, ss)):
                print(f"    Peak {i+1: 3.0f}    {lw: 2.2e} | {m: 2.2e} | {s: 2.2e}")

        return lws, ms, ss

    def gen_mas1_params(self, n):
        """
        Generate MAS parameters (rate, broadening, shift)

        Input:      - n     Number of peaks in the spectrum

        Outputs:    - wr    MAS rates
                    - lws    MAS-dependent GLS broadening for each peak
                    - ms    MAS-dependent GLS mixing for each peak
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

        if isinstance(self.mas_lw_range[0], list):
            # Randomly select broadening range
            lws = []
            for _ in range(n):
                # Randomly select broadening range
                i = self.select_index_with_prob(self.mas_lw_p)
                # Generate MAS-dependent GLS broadening (uniform distribution)
                dlw = self.mas_lw_range[i][1] - self.mas_lw_range[i][0]
                lw0 = self.mas_lw_range[i][0]
                lws.append(lw0 + (np.random.rand() * dlw))
            lws = np.array(lws)

        else:
            dlw = self.mas_lw_range[1] - self.mas_lw_range[0]
            lw0 = self.mas_lw_range[0]
            lws = lw0 + (np.random.rand(n) * dlw)

        if isinstance(self.mas_m_range[0], list):
            ms = []
            for _ in range(n):
                # Randomly select GLS mixing range
                i = self.select_index_with_prob(self.mas_m_p)
                # Generate MAS-dependent GLS mixing (uniform distribution)
                dm = self.mas_m_range[i][1] - self.mas_m_range[i][0]
                m0 = self.mas_m_range[i][0]
                ms.append(m0 + (np.random.rand() * dm))
            ms = np.array(ms)

        else:
            # Generate MAS-dependent GLS mixing (uniform distribution)
            dm = self.mas_m_range[1] - self.mas_m_range[0]
            m0 = self.mas_m_range[0]
            ms = m0 + (np.random.rand(n) * dm)

        if isinstance(self.mas_s_range[0], list):
            ss = []
            for _ in range(n):
                # randomly select shift range
                i = self.select_index_with_prob(self.mas_s_p)
                # Generate MAS-dependent shift
                ds = self.mas_s_range[i][1] - self.mas_s_range[i][0]
                s0 = self.mas_s_range[i][0]
                ss.append(s0 + (np.random.rand() * ds))
            ss = np.array(ss)

        else:
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
            print("                lw [Hz^2] |     m     |  s [Hz^2]")
            for i, (lw, m, s) in enumerate(zip(lws, ms, ss)):
                print(f"    Peak {i+1: 3.0f}    {lw: 2.2e} | {m: 2.2e} | {s: 2.2e}")

        return wr, lws, ms, ss, ps

    def gen_mas2_params(self, n):
        """
        Generate second-order MAS parameters (rate, broadening, shift)

        Input:      - n     Number of peaks in the spectrum

        Outputs:    - ws    MAS^2-dependent GLS broadening for each peak
                    - ms    MAS^2-dependent GLS mixing for each peak
                    - ss    MAS^2-dependent shift for each peak
        """

        if isinstance(self.mas2_lw_range[0], list):
            # Randomly select broadening range
            lws2 = []
            for _ in range(n):
                # Randomly select broadening range
                i = self.select_index_with_prob(self.mas2_lw_p)

                # Generate MAS-dependent Lorentzian broadening (uniform distribution)
                dlw = self.mas2_lw_range[i][1] - self.mas2_lw_range[i][0]
                lw0 = self.mas2_lw_range[i][0]
                lws2.append(lw0 + (np.random.rand() * dlw))
            lws2 = np.array(lws2)

        else:
            # Generate MAS-dependent Lorentzian broadening (uniform distribution)
            dlw = self.mas2_lw_range[1] - self.mas2_lw_range[0]
            lw0 = self.mas2_lw_range[0]
            lws2 = lw0 + (np.random.rand(n) * dlw)


        if isinstance(self.mas2_m_range[0], list):

            ms2 = []
            for _ in range(n):
                # Randomly select mixing range
                i = self.select_index_with_prob(self.mas2_m_p)

                # Generate MAS-dependent Gaussian broadening (uniform distribution)
                dm = self.mas2_m_range[i][1] - self.mas2_m_range[i][0]
                m0 = self.mas2_m_range[i][0]
                ms2.append(m0 + (np.random.rand() * dm))
            ms2 = np.array(ms2)

        else:
            # Generate MAS-dependent Gaussian broadening (uniform distribution)
            dm = self.mas2_m_range[1] - self.mas2_m_range[0]
            m0 = self.mas2_m_range[0]
            ms2 = m0 + (np.random.rand(n) * dm)

        if isinstance(self.mas2_s_range[0], list):

            ss2 = []
            for _ in range(n):
                # Randomly select shift range
                i = self.select_index_with_prob(self.mas2_s_p)

                # Generate MAS-dependent shift
                ds = self.mas2_s_range[i][1] - self.mas2_s_range[i][0]
                s0 = self.mas2_s_range[i][0]
                ss2.append(s0 + (np.random.rand() * ds))
            ss2 = np.array(ss2)

        else:
            # Generate MAS-dependent shift
            ds = self.mas2_s_range[1] - self.mas2_s_range[0]
            s0 = self.mas2_s_range[0]
            ss2 = s0 + (np.random.rand(n) * ds)

        inds = np.random.rand(n) >= self.mas_w2_p
        lws2[inds] = 0.
        ms2[inds] = 0.
        ss2[inds] = 0.

        if self.debug:
            print("\n  Generating MAS^2-dependent parameters...")
            print("                lw [Hz^2] |     m     |  s [Hz^2]")
            for i, (lw, m, s) in enumerate(zip(lws2, ms2, ss2)):
                print(f"    Peak {i+1: 3.0f}    {lw: 2.2e} | {m: 2.2e} | {s: 2.2e}")

        return lws2, ms2, ss2

    def gls(self, x, p, w, m):
        """
        Gaussial-Lorentzian sum broadening function

        Inputs: - x     Array on which the function should be evaluated
                - p     Peak position
                - w     Width of the function
                - m     Mixing paramer (w=0: pure Gaussian, w=1: pure Lorentzian)
                - h     Height of the function

        Output: - y     GLS function evaluated on the array x
        """

        if m > 1.:
            m = 1.
        if m < 0.:
            m = 0.

        y = (1-m) * np.exp(-4 * np.log(2) * np.square(x-p) / (w ** 2))

        y += m / (1 + 4 * np.square(x-p) / (w ** 2))

        return y / np.sum(y)

    def mas_broaden(self, specs, wr, lws0, lws, ms0, ms, ss0, ss, ps, ms_other):
        """
        Broaden isotropic spectra with MAS-dependent parameters

        Inputs: - specs         Isotropic spectrum for each peak
                - wr            MAS rates
                - lws            MAS-dependent GLS broadening for each peak
                - ms            MAS-dependent GLS mixing for each peak
                - ss            MAS-dependent shift for each peak
                - ps            MAS-dependent phase for each spectrum/peak

        Output: - brd_specs     Broadened spectra
        """

        n_mas = wr.shape[0]
        n_pks, n_pts = specs.shape
        r = np.arange(1., n_pts+1) / n_pts

        data = np.empty((n_mas, n_pks, n_pts), dtype=complex)

        if not self.peakwise_phase:
            raise NotImplementedError()

        for i, (w, p, m_other) in enumerate(zip(wr, ps, ms_other)):

            for j, (spec, lw0, lw, m0, m, s0, s, pi, mi) in enumerate(zip(specs, lws0, lws, ms0, ms, ss0, ss, p, m_other)):

                if self.mas_s_noise > 0:
                    ds = np.random.randn() * self.mas_s_noise
                    brd_fid = np.fft.ifft(spec) * np.exp(1j * 2 * np.pi * (s0 + s / w + ds) * self.t)
                else:
                    brd_fid = np.fft.ifft(spec) * np.exp(1j * 2 * np.pi * (s0 + s / w) * self.t)

                if np.abs(pi) > 0.:
                    brd_fid *= np.exp(-1j * pi)
                data[i, j] = np.fft.fft(brd_fid)

                lw_noise = np.random.randn() * self.mas_lw_noise * (lw / wr[0] - lw / wr[-1])

                if mi >= 0:
                    data[i, j] = np.convolve(data[i, j], self.gls(self.f, self.f[-1] / 2., lw_noise + lw0 + lw / w, mi), mode="same")
                else:
                    data[i, j] = np.convolve(data[i, j], self.gls(self.f, self.f[-1] / 2., lw_noise + lw0 + lw / w, m0 + m / w), mode="same")

        if self.encode_imag:
            output = np.empty((n_mas, 2, n_pts))
            output[:, 0, :] = np.mean(np.real(data), axis=1)
            output[:, 1, :] = np.mean(np.imag(data), axis=1)

        else:
            output = np.empty((n_mas, 1, n_pts))
            output[:, 0, :] = np.mean(np.real(data), axis=1)

        return output

    def mas2_broaden(self, specs, wr, lws0, lws, lws2, ms0, ms, ms2, ss0, ss, ss2, ps, ms_other):
        """
        Broaden isotropic spectra with MAS^2-dependent parameters

        Inputs: - specs         Isotropic spectrum for each peak
                - wr            MAS rates
                - lws            MAS-dependent GLS broadening for each peak
                - lws2           MAS-dependent GLS broadening for each peak
                - ms            MAS-dependent GLS mixing for each peak
                - ms2           MAS-dependent GLS mixing for each peak
                - ss            MAS-dependent shift for each peak
                - ss2           MAS-dependent shift for each peak
                - ps            MAS-dependent phase for each spectrum/peak

        Output: - brd_specs     Broadened spectra
        """

        n_mas = wr.shape[0]
        n_pks, n_pts = specs.shape
        r = np.arange(1., n_pts+1) / n_pts

        data = np.empty((n_mas, n_pks, n_pts), dtype=complex)

        for i, (w, p, m_other) in enumerate(zip(wr, ps, ms_other)):

            for j, (spec, lw0, lw, lw2, m0, m, m2, s0, s, s2, pi, mi) in enumerate(zip(specs, lws0, lws, lws2, ms0, ms, ms2, ss0, ss, ss2, p, m_other)):

                if self.mas_s_noise > 0:
                    ds = np.random.randn() * self.mas_s_noise
                    brd_fid = np.fft.ifft(spec) * np.exp(1j * 2 * np.pi * (s0 + s / w + s2 / (w ** 2) + ds) * self.t)
                else:
                    brd_fid = np.fft.ifft(spec) * np.exp(1j * 2 * np.pi * (s0 + s / w + s2 / (w ** 2)) * self.t)

                if np.abs(pi) > 0.:
                    brd_fid *= np.exp(-1j * pi)
                data[i, j] = np.fft.fft(brd_fid)

                lw_noise = np.random.randn() * self.mas_lw_noise * (lw / wr[0] - lw / wr[-1] + lw2 / (wr[0]**2) - lw2 / (wr[-1]**2))

                if mi >= 0:
                    data[i, j] = np.convolve(data[i, j], self.gls(self.f, self.f[-1] / 2., lw_noise + lw0 + lw / w + lw2 / (w ** 2), mi), mode="same")
                else:
                    data[i, j] = np.convolve(data[i, j], self.gls(self.f, self.f[-1] / 2.,
                                                                  lw_noise + lw0 + lw / w + lw2 / (w ** 2),
                                                                  m0 + m / w + m2 / (w ** 2)), mode="same")

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
        int0 = np.sum(np.real(iso))
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

        return np.real(iso), np.real(specs), brd_specs

    def finalize_spectra(self, iso, specs, brd_specs, wr):

        iso = np.real(iso)
        if self.positive:
            iso[iso < 0.] = 0.

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

    def gen_mas_mixing(self, n, wr):
        ms = -1 * np.ones((len(wr), n))

        dm = self.mas_other_mixing_range[1] - self.mas_other_mixing_range[0]
        m0 = self.mas_other_mixing_range[0]

        for i in range(n):
            p = np.random.random()
            if p < self.mas_other_mixing_p:
                j = self.select_index_with_prob(self.mas_other_mixing_probs)

                if self.mas_other_mixing_trends[j] == "constant":
                    ms[:, i] = m0 + np.random.rand() * dm


                elif self.mas_other_mixing_trends[j] == "increase":
                    m1 = m0 + np.random.rand() * dm
                    m2 = m0 + np.random.rand() * dm
                    m_min = min(m1, m2)
                    m_max = max(m1, m2)
                    gen_dm = m_max - m_min
                    gen_ms = m_min + np.random.rand(len(wr)) * gen_dm
                    ms[:, i] = np.sort(gen_ms)

                elif self.mas_other_mixing_trends[j] == "decrease":
                    m1 = m0 + np.random.rand() * dm
                    m2 = m0 + np.random.rand() * dm
                    m_min = min(m1, m2)
                    m_max = max(m1, m2)
                    gen_dm = m_max - m_min
                    gen_ms = m_min + np.random.rand(len(wr)) * gen_dm
                    ms[:, i] = np.sort(gen_ms)[::-1]

                else:
                    raise ValueError(f"Unknown trend: {self.mas_other_mixing_trends[j]}")

        return ms

    def __getitem__(self, _):
        """
        Generate an input
        """

        # Generate isotropic spectrum
        n, specs, iso = self.gen_iso_spectra()

        lws0, ms0, ss0 = self.gen_mas0_params(n)

        # Generate MAS-dependent parameters
        wr, lws, ms, ss, ps = self.gen_mas1_params(n)

        # Generate MAS-dpendent mixing
        ms_other = self.gen_mas_mixing(n, wr)

        if self.mas_w2 and np.random.random() < self.mas_w2_p:
            lws2, ms2, ss2 = self.gen_mas2_params(n)
            # Broaden isotropic spectrum with MAS-dependent parameters
            brd_specs = self.mas2_broaden(specs, wr, lws0, lws, lws2, ms0, ms, ms2, ss0, ss, ss2, ps, ms_other)
        else:
            # Broaden isotropic spectrum with MAS-dependent parameters
            brd_specs = self.mas_broaden(specs, wr, lws0, lws, ms0, ms, ss0, ss, ps, ms_other)

        # Set the minimum of each spectrum to zero
        brd_specs -= np.min(brd_specs[:, 0], axis=1)[:, np.newaxis, np.newaxis]

        # Normalize spectra
        iso, specs, brd_specs = self.normalize_spectra(iso, specs, brd_specs)

        return self.finalize_spectra(iso, specs, brd_specs, wr)
