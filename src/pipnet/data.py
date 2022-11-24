###########################################################################
###                               PIPNet                                ###
###                      Data generation functions                      ###
###                        Author: Manuel Cordova                       ###
###                       Last edited: 2022-09-26                       ###
###########################################################################

import numpy as np
import torch
import scipy as sp



def select_index_with_prob(prob):

    # Randomly select range
    p = np.random.rand()
    i = 0
    p_tot = prob[i]
    while p > p_tot:
        i += 1
        p_tot += prob[i]
    return i

class IsoGenerator():
    """
    Isotropic spectra generator
    """

    def __init__(
        self,
        td=256,
        Fs=12800,
        nmin=1,
        nmax=1,
        freq_range=[2000., 10000.],
        positive=True,
        gmin=1,
        gmax=1,
        spread=0.03,
        lw_range=[[5e1, 2e2]],
        lw_probs=[1.],
        int_range=[0.5, 1.],
        norm_height=False,
        phase=0.,
        debug=False,
    ):
        """
        Inputs: - td                Number of points in time domain
                - Fs                Sampling frequency
                - nmin              Minimum number of peaks in an isotropic spectrum
                - nmax              Maximum number of peaks in an isotropic spectrum
                - shift_range       Allowed frequency range for peaks
                - positive          Remove negative parts of isotropic spectra and make them zero
                - gmin              Minimum number of Gaussians in an isotropic peak
                - gmax              Maximum number of Gaussians in an isotropic peak
                - spread            Spread between Gaussian frequencies within an isotropic peak
                - lw_range          Linewidth ranges for Gaussians
                - lw_probs          Probability of each linewidth range
                - debug             Print additional debug informations
        """

        self.td = td
        self.Fs = Fs
        self.nmin = nmin
        self.nmax = nmax
        self.freq_range = freq_range
        self.positive = positive
        self.gmin = gmin
        self.gmax = gmax
        self.spread = spread
        self.lw_range = lw_range
        self.lw_probs = lw_probs
        self.int_range = int_range
        self.norm_height = norm_height
        self.phase = phase
        self.debug = debug

        # Get time domain points
        self.t = np.arange(0, self.td, 1) / self.Fs

        # Get frequency domain points
        self.df = self.Fs / self.td
        self.f = np.arange(0, self.td, 1) * self.df

        return
    


    def _gen_iso_peak(self):
        """
        Generate an isotropic peak as a sum of Gaussians
        """

        # Randomly get the number of Gaussians in a peak (uniform distribution)
        dn = self.gmax - self.gmin + 1
        n = self.gmin + int(np.random.rand() * dn)

        # Get isotropic frequency
        dw0 = self.freq_range[1] - self.freq_range[0]
        w0 = self.freq_range[0] + (np.random.rand() * dw0)

        # Get frequency displacement for each Gaussian (normal distribution)
        w0s = w0 + (np.random.randn(n) * self.spread)

        # Get broadening for each Gaussian (uniform distribution)
        if isinstance(self.lw_range[0], list):
            i = select_index_with_prob(self.lw_probs)
            dlw = self.lw_range[i][1] - self.lw_range[i][0]
            lws = self.lw_range[i][0] + np.random.rand(n) * dlw

        else:
            dlw = self.lw_range[1]-self.lw_range[0]
            lws = self.lw_range[0] + (np.random.rand(n) * dlw)

        # Get phase for each Gaussian (normal distribution)
        if self.phase > 0.:
            ps = np.random.randn(n) * self.phase
        else:
            ps = np.zeros(n)

        if self.debug:
            print(f"  Generating a peak of {n} Gaussians around {w0} Hz...")
            print("           v[Hz] | dv [Hz] | phi [rad]")
            for i, (w0, lw, p) in enumerate(zip(w0s, lws, ps)):
                print(f"    G {i+1: 2.0f} | {w0:.3f} | {lw:.5f} | {p:.7f}")

        # Build FID
        fid = np.zeros(len(self.t), dtype=complex)
        fs = []

        for w0, lw, p in zip(w0s, lws, ps):
            # Frequency
            f = np.exp(w0 * self.t * 2 * np.pi * 1j)
            # Broadening
            f *= np.exp(-1 * (self.t * lw) ** 2)
            # Phase
            f *= np.exp(-1j * p)
            f[0] /= 2
            fs.append(f)
            fid += f

        fid /= n

        return fid, fs
    


    def _gen_iso_spectra(self):
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
            fids[i], _ = self._gen_iso_peak()

        # Fourier transform
        isos = np.fft.fft(fids, axis=1)

        # Normalize by height instead of integral
        if self.norm_height:
            isos /= np.max(np.real(isos), axis=1)[:, np.newaxis]

        # Randomize intensity
        da = self.int_range[1] - self.int_range[0]
        a = self.int_range[0] + (np.random.rand(n) * da)
        isos *= a[:, np.newaxis]

        iso = np.expand_dims(np.sum(isos, axis=0), 0)

        return n, isos, iso
    


    def __call__(self):
        """
        Generate isotropic spectra
        """
        return self._gen_iso_spectra()



class MASGenerator():
    """
    MAS-dependent parameters generation
    """
    def __init__(
        self,
        nw=8,
        mas_w_range=[],
        random_mas=True,
        mas1_lw_range=[],
        mas1_lw_probs=[],
        mas1_m_range=[],
        mas1_m_probs=[],
        mas1_s_range=[],
        mas1_s_probs=[],
        mas2_prob=1.,
        mas2_lw_range=[],
        mas2_lw_probs=[],
        mas2_m_range=[],
        mas2_m_probs=[],
        mas2_s_range=[],
        mas2_s_probs=[],
        non_mas_p=0.,
        non_mas_m_trends=[],
        non_mas_m_probs=[],
        non_mas_m_range=[],
        mas_phase_scale=0.,
        mas_phase_p=0.,
        int_decrease_p=0.,
        int_decrease_scale=0.1,
        debug=False,
    ):
        """
        Inputs: - nw                    Number of MAS rates to generate
                - mas_w_range           Range of MAS rates to generate
                - mas1_lw_range         Linewidth ranges for MAS-dependent broadening
                - mas1_lw_probs         Probability of each linewidth range for MAS-dependent broadening
                - mas1_m_range          Mixing ranges for MAS-dependent broadening
                - mas1_m_probs          Probability of each mixing range for MAS-dependent broadening
                - mas1_s_range          Shift ranges for MAS-dependent shift
                - mas1_s_probs          Probability of each shift range for MAS-dependent shift
                - mas2_prob             Probability of MAS2 dependency
                - mas2_lw_range         Linewidth ranges for MAS2-dependent broadening
                - mas2_lw_probs         Probability of each linewidth range for MAS2-dependent broadening
                - mas2_m_range          Mixing ranges for MAS2-dependent broadening
                - mas2_m_probs          Probability of each mixing range for MAS2-dependent broadening
                - mas2_s_range          Shift ranges for MAS2-dependent shift
                - mas2_s_probs          Probability of each shift range for MAS2-dependent shift
                - non_mas_m_trends      Trends for other mixing MAS-dependency
                - non_mas_m_probs       Probability of each trend for other mixing MAS-dependency
                - non_mas_m_range       Allowed range of mixing values for other mixing MAS-dependency
        """

        self.nw = nw
        self.mas_w_r = mas_w_range
        self.random_mas = random_mas
        self.mas_phase_scale = mas_phase_scale
        self.mas_phase_p = mas_phase_p

        self.mas1_lw_r = mas1_lw_range
        self.mas1_lw_p = mas1_lw_probs
        self.mas1_m_r = mas1_m_range
        self.mas1_m_p = mas1_m_probs
        self.mas1_s_r = mas1_s_range
        self.mas1_s_p = mas1_s_probs

        self.mas2_p = mas2_prob
        self.mas2_lw_r = mas2_lw_range
        self.mas2_lw_p = mas2_lw_probs
        self.mas2_m_r = mas2_m_range
        self.mas2_m_p = mas2_m_probs
        self.mas2_s_r = mas2_s_range
        self.mas2_s_p = mas2_s_probs

        self.non_mas_p = non_mas_p
        self.non_mas_m_trends = non_mas_m_trends
        self.non_mas_m_p = non_mas_m_probs
        self.non_mas_m_r = non_mas_m_range
        
        self.int_decrease_p = int_decrease_p
        self.int_decrease_scale = int_decrease_scale

        self.debug = debug

        return
    


    def _gen_mas1_params(self, n):
        
        # Generate MAS-dependent linewidth parameters
        if isinstance(self.mas1_lw_r[0], list):
            # Randomly select broadening range
            lws = []
            for _ in range(n):
                # Randomly select broadening range
                i = select_index_with_prob(self.mas1_lw_p)
                # Generate MAS-dependent GLS broadening (uniform distribution)
                dlw = self.mas1_lw_r[i][1] - self.mas1_lw_r[i][0]
                lw0 = self.mas1_lw_r[i][0]
                lws.append(lw0 + (np.random.rand() * dlw))
            lws = np.array(lws)

        else:
            dlw = self.mas1_lw_r[1] - self.mas1_lw_r[0]
            lw0 = self.mas1_lw_r[0]
            lws = lw0 + (np.random.rand(n) * dlw)

        # Generate MAS-dependent mixing parameters
        if isinstance(self.mas1_m_r[0], list):
            ms = []
            for _ in range(n):
                # Randomly select GLS mixing range
                i = select_index_with_prob(self.mas1_m_p)
                # Generate MAS-dependent GLS mixing (uniform distribution)
                dm = self.mas1_m_r[i][1] - self.mas1_m_r[i][0]
                m0 = self.mas1_m_r[i][0]
                ms.append(m0 + (np.random.rand() * dm))
            ms = np.array(ms)

        else:
            dm = self.mas1_m_r[1] - self.mas1_m_r[0]
            m0 = self.mas1_m_r[0]
            ms = m0 + (np.random.rand(n) * dm)

        # Generate MAS-dependent shift parameters
        if isinstance(self.mas1_s_r[0], list):
            ss = []
            for _ in range(n):
                # randomly select shift range
                i = select_index_with_prob(self.mas1_s_p)
                # Generate MAS-dependent shift
                ds = self.mas1_s_r[i][1] - self.mas1_s_r[i][0]
                s0 = self.mas1_s_r[i][0]
                ss.append(s0 + (np.random.rand() * ds))
            ss = np.array(ss)

        else:
            ds = self.mas1_s_r[1] - self.mas1_s_r[0]
            s0 = self.mas1_s_r[0]
            ss = s0 + (np.random.rand(n) * ds)

        if self.debug:
            print("\n  Generating first-order MAS-dependent parameters...")
            print("                lw [Hz^2] |     m     |  s [Hz^2]")
            for i, (lw, m, s) in enumerate(zip(lws, ms, ss)):
                print(f"    Peak {i+1: 3.0f}    {lw: 2.2e} | {m: 2.2e} | {s: 2.2e}")

        return lws, ms, ss
    


    def _gen_mas2_params(self, n):
        
        # Generate MAS-dependent linewidth parameters
        if isinstance(self.mas2_lw_r[0], list):
            # Randomly select broadening range
            lws = []
            for _ in range(n):
                # Randomly select broadening range
                i = select_index_with_prob(self.mas2_lw_p)
                # Generate MAS-dependent GLS broadening (uniform distribution)
                dlw = self.mas2_lw_r[i][1] - self.mas2_lw_r[i][0]
                lw0 = self.mas2_lw_r[i][0]
                lws.append(lw0 + (np.random.rand() * dlw))
            lws = np.array(lws)

        else:
            dlw = self.mas2_lw_r[1] - self.mas2_lw_r[0]
            lw0 = self.mas2_lw_r[0]
            lws = lw0 + (np.random.rand(n) * dlw)

        # Generate MAS-dependent mixing parameters
        if isinstance(self.mas2_m_r[0], list):
            ms = []
            for _ in range(n):
                # Randomly select GLS mixing range
                i = select_index_with_prob(self.mas2_m_p)
                # Generate MAS-dependent GLS mixing (uniform distribution)
                dm = self.mas2_m_r[i][1] - self.mas2_m_r[i][0]
                m0 = self.mas2_m_r[i][0]
                ms.append(m0 + (np.random.rand() * dm))
            ms = np.array(ms)

        else:
            dm = self.mas2_m_r[1] - self.mas2_m_r[0]
            m0 = self.mas2_m_r[0]
            ms = m0 + (np.random.rand(n) * dm)

        # Generate MAS-dependent shift parameters
        if isinstance(self.mas2_s_r[0], list):
            ss = []
            for _ in range(n):
                # randomly select shift range
                i = select_index_with_prob(self.mas2_s_p)
                # Generate MAS-dependent shift
                ds = self.mas2_s_r[i][1] - self.mas2_s_r[i][0]
                s0 = self.mas2_s_r[i][0]
                ss.append(s0 + (np.random.rand() * ds))
            ss = np.array(ss)

        else:
            ds = self.mas2_s_r[1] - self.mas2_s_r[0]
            s0 = self.mas2_s_r[0]
            ss = s0 + (np.random.rand(n) * ds)

        if self.debug:
            print("\n  Generating second-order MAS-dependent parameters...")
            print("                lw [Hz^3] |     m     |  s [Hz^3]")
            for i, (lw, m, s) in enumerate(zip(lws, ms, ss)):
                print(f"    Peak {i+1: 3.0f}    {lw: 2.2e} | {m: 2.2e} | {s: 2.2e}")

        return lws, ms, ss
    


    def _gen_non_mas_mixing(self, n):

        ms = -1 * np.ones((self.nw, n))

        dm = self.non_mas_m_r[1] - self.non_mas_m_r[0]
        m0 = self.non_mas_m_r[0]

        for i in range(n):
            p = np.random.rand()
            if p < self.non_mas_p:
                j = select_index_with_prob(self.non_mas_m_p)

                if self.non_mas_m_trends[j] == "constant":
                    ms[:, i] = m0 + np.random.rand() * dm


                elif self.non_mas_m_trends[j] == "increase":
                    m1 = m0 + np.random.rand() * dm
                    m2 = m0 + np.random.rand() * dm
                    gen_dm = m2 - m1
                    gen_ms = m1 + np.random.rand(self.nw) * gen_dm
                    ms[:, i] = np.sort(gen_ms)

                elif self.non_mas_m_trends[j] == "decrease":
                    m1 = m0 + np.random.rand() * dm
                    m2 = m0 + np.random.rand() * dm
                    gen_dm = m2 - m1
                    gen_ms = m1 + np.random.rand(self.nw) * gen_dm
                    ms[:, i] = np.sort(gen_ms)[::-1]

                else:
                    raise ValueError(f"Unknown trend: {self.non_mas_m_trends[j]}")

        return ms
    


    def __call__(self, n, ws=None):

        # Generate MAS rates
        if ws is None:
            dw = self.mas_w_r[1] - self.mas_w_r[0]
            w0 = self.mas_w_r[0]
            if self.random_mas:
                ws = w0 + dw * np.sort(np.random.rand(self.nw))
            else:
                ws = np.linspace(self.mas_w_r[0], self.mas_w_r[1], self.nw)
        
        # Randomly phase each peak in the MAS spectra (normal distribution)
        ps = np.zeros((self.nw, n))
        for i in range(n):
            if np.random.rand() < self.mas_phase_p:
                ps[:, i] = np.random.randn(self.nw) * self.mas_phase_scale

        # Generate intensity reduction for randomly selected peaks
        hs = np.ones((self.nw, n))
        for i in range(n):
            if np.random.rand() < self.int_decrease_p:
                s = np.random.rand() * (self.int_decrease_scale[1] - self.int_decrease_scale[0]) + self.int_decrease_scale[0]
                hs[:, i] = 1 - (np.arange(self.nw) * s / self.nw)

        if self.debug:
            print("\n  Generated MAS rates: " + ", ".join([f"{w:6.0f}" for w in ws]) + " Hz")
            print("  Generated MAS phases:")
            for i, p in enumerate(ps.T):
                print(f"    Peak {i+1: 3.0f}           " + ", ".join([f"{pi: 2.3f}" for pi in p]) + " rad")

        ls1, ms1, ss1 = self._gen_mas1_params(n)

        if np.random.rand() < self.mas2_p:
            ls2, ms2, ss2 = self._gen_mas2_params(n)
        else:
            ls2 = np.zeros_like(ls1)
            ms2 = np.zeros_like(ms1)
            ss2 = np.zeros_like(ss1)
        
        ms0 = self._gen_non_mas_mixing(n)

        return ws, ls1, ls2, ms0, ms1, ms2, ss1, ss2, ps, hs



class Dataset(torch.utils.data.Dataset):
    """
    PIP dataset with GLS broadening
    """

    def __init__(
        self,
        iso_pars,
        mas_pars,
        positive_iso=False,
        encode_imag=False,
        encode_wr=True,
        noise=0.,
        mas_l_noise=0.,
        mas_s_noise=0.,
        smooth_end_len=10,
        iso_spec_norm=256.,
        mas_spec_norm=64.,
        wr_norm_factor=100_000.,
        wr_inv=False,
        gen_mas_shifts=False,
        debug=False,
    ):
        """
        Inputs: - iso_pars          Isotropic parameters
                - mas_pars          MAS parameters
                - encode_imag       Encode the imaginary part of the spectra
                - encode_mas        Encode the MAS rate
                - noise             Add noise to the MAS spectra
                - smooth_end_len    Length over which the spectra should be smoothed to zero
                - iso_spec_norm     Number to divide the isotropic spectra by for normalization
                - mas_spec_norm     Number to divide the MAS spectra by for normalization
                - wr_norm_factor    Number to divide the MAS rate by for normalization
                - wr_inv            Encode the inverse MAS rate
        """

        super(Dataset, self).__init__()

        self.gen_iso = IsoGenerator(**iso_pars)
        self.gen_mas = MASGenerator(**mas_pars)

        self.t = self.gen_iso.t
        self.f = self.gen_iso.f
        self.nw = self.gen_mas.nw
        self.nmax = self.gen_iso.nmax

        self.positive_iso = positive_iso
        self.encode_im = encode_imag
        self.encode_wr = encode_wr
        self.noise = noise
        self.mas_l_noise = mas_l_noise
        self.mas_s_noise = mas_s_noise
        self.l_smooth = smooth_end_len
        self.norm_iso = iso_spec_norm
        self.norm_mas = mas_spec_norm
        self.norm_wr = wr_norm_factor
        self.wr_inv = wr_inv
        self.gen_mas_shifts = gen_mas_shifts

        return



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
        
        if w < 1.:
            w = 1.

        y = (1-m) * np.exp(-4 * np.log(2) * np.square(x-p) / (w ** 2))

        y += m / (1 + 4 * np.square(x-p) / (w ** 2))

        return y / np.sum(y)



    def mas_broaden(self, specs, ws, ls1, ls2, ms0, ms1, ms2, ss1, ss2, ps, hs):
        """
        Broaden isotropic spectra with MAS-dependent parameters

        Inputs: - specs         Isotropic spectrum for each peak
                - wr            MAS rates
                - lws            MAS-dependent GLS broadening for each peak
                - ms            MAS-dependent GLS mixing for each peak
                - ss            MAS-dependent shift for each peak
                - ps            MAS-dependent phase for each spectrum/peak
                - hs            MAS-dependent intensities

        Output: - brd_specs     Broadened spectra
        """

        n_pks, n_pts = specs.shape

        data = np.empty((self.nw, n_pks, n_pts), dtype=complex)

        for i, (w, p, m, h) in enumerate(zip(ws, ps, ms0, hs)):

            w2 = w ** 2

            for j, (spec, l1, l2, m0, m1, m2, s1, s2, p0, hi) in enumerate(zip(specs, ls1, ls2, m, ms1, ms2, ss1, ss2, p, h)):

                # Apply shift
                ds = np.random.randn() * self.mas_s_noise
                if self.gen_mas_shifts:
                    if w == ws[-1]:
                        ds = 0.
                    ds -= s1 / ws[-1] + s2 / (ws[-1]**2)
                brd_fid = np.fft.ifft(spec) * np.exp(1j * 2 * np.pi * ((s1 / w) + (s2 / w2) + ds) * self.t)

                # Apply phase
                brd_fid *= np.exp(-1j * p0)
                data[i, j] = np.fft.fft(brd_fid)

                # Apply broadening
                dl = np.random.randn() * self.mas_l_noise * (l1 / ws[0] + l2 / (ws[0]**2) - l1 / ws[-1] - l2 / (ws[-1]**2))
                if m0 >= 0:
                    data[i, j] = np.convolve(
                        data[i, j],
                        self.gls(self.f, self.f[-1]/2., max(0., dl + l1 / w + l2 / w2), m0),
                        mode="same"
                    ) * hi
                else:
                    data[i, j] = np.convolve(
                        data[i, j],
                        self.gls(self.f, self.f[-1]/2., max(0., dl + l1 / w + l2 / w2), m1 / w + m2 / w2),
                        mode="same"
                    ) * hi

        if self.encode_im:
            output = np.empty((self.nw, 2, n_pts))
            output[:, 0, :] = np.sum(np.real(data), axis=1)
            output[:, 1, :] = np.sum(np.imag(data), axis=1)

        else:
            output = np.empty((self.nw, 1, n_pts))
            output[:, 0, :] = np.sum(np.real(data), axis=1)

        return output



    def normalize_spectra(self, iso, specs, brd_specs):

        # Normalize spectra
        iso /= self.norm_iso
        specs /= self.norm_iso
        brd_specs /= self.norm_mas

        return np.real(iso), np.real(specs), brd_specs



    def finalize_spectra(self, iso, specs, brd_specs, ws, return_ws=False):

        # Set isotropic spectra to positive values
        iso = np.real(iso)
        specs = np.real(specs)
        if self.positive_iso:
            iso[iso < 0.] = 0.
            specs[specs < 0.] = 0.

        # Add noise
        if self.noise > 0.:
            brd_specs += np.random.normal(loc=0., scale=self.noise, size=brd_specs.shape)

        # Smooth spectra edges
        if self.l_smooth > 0:
            for i in range(self.l_smooth):
                brd_specs[:, :, i] *= i / self.l_smooth
                brd_specs[:, :, -(i+1)] *= i / self.l_smooth

        # Encode MAS rate
        if self.encode_wr:
            ws_enc = np.repeat(np.expand_dims(ws, (1, 2)), brd_specs.shape[-1], axis=2)

            # Normalize MAS rate
            ws_enc /= self.norm_wr
            if self.wr_inv:
                ws_enc = 1. / ws_enc
            
            # Append encoded MAS rate to spectra
            brd_specs = np.append(brd_specs, ws_enc, axis=1)

        # Add dummy single-peak isotropic spectra to get the same array shape for all data generation
        if specs.shape[0] < self.nmax:
            specs = np.pad(specs, ((0, self.nmax - specs.shape[0]), (0, 0)))
        
        if return_ws:
            return (torch.from_numpy(brd_specs.astype(np.float32)),
                    torch.from_numpy(specs.astype(np.float32)),
                    torch.from_numpy(iso.astype(np.float32)),
                    ws, ws_enc[:, 0, 0])
        else:
            return (torch.from_numpy(brd_specs.astype(np.float32)),
                    torch.from_numpy(specs.astype(np.float32)),
                    torch.from_numpy(iso.astype(np.float32)))
    


    def generate_batch(self, size=4):
        """
        Generate a batch of data:

        Input:      - size  Batch size

        Outputs:    - X     Batch of inputs
                    - y     Batch of outputs
        """

        X = []
        y = []

        for i in range(size):
            Xi, _, yi = self.__getitem__(0)
            X.append(Xi.unsqueeze(0))
            y.append(yi.unsqueeze(0))

        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)

        return X, y


    
    def __len__(self):
        """
        Dummy function for pytorch to work properly
        """

        return int(1e12)
    


    def __getitem__(self, _, ws=None, return_ws=False):
        """
        Generate an input
        """

        # Generate isotropic spectrum
        n, specs, iso = self.gen_iso()
        ws, ls1, ls2, ms0, ms1, ms2, ss1, ss2, ps, hs = self.gen_mas(n, ws=ws)

        brd_specs = self.mas_broaden(specs, ws, ls1, ls2, ms0, ms1, ms2, ss1, ss2, ps, hs)

        # Normalize spectra
        iso, specs, brd_specs = self.normalize_spectra(iso, specs, brd_specs)

        return self.finalize_spectra(iso, specs, brd_specs, ws, return_ws=return_ws)



class Dataset2D(torch.utils.data.Dataset):
    """
    """
    def __init__(
        self,
        params_x,
        params_y,
        rot_prob=0.5,
        rot_range=[0., 90.],
        noise=0.,
    ):

        super(Dataset2D, self).__init__()
        
        self.xgen = Dataset(**params_x)
        self.ygen = Dataset(**params_y)

        self.prot = rot_prob
        self.rrot = rot_range
        self.noise = noise

        return
    


    def make_2d(self, X, Y, ws=None):
        """
        Make 2D spectrum from two 1D spectra

        Inputs: - X     First 1D spectrum or batch of spectra
                - Y     Second 1D spectrum or batch of spectra
                - ws    Array of MAS rates corresponding to each 1D spectrum

        Output: - Z     Output 2D spectrum or batch of spectra
        """

        if X.ndim == 3 and Y.ndim == 3:
            
            Z = torch.einsum("wcp,wcq->wcqp", X, Y)
            
            # Encode MAS rate
            if ws is not None:
                W = np.expand_dims(ws, (1, 2))
                W = np.repeat(W, Z.shape[-2], axis=1)
                W = np.repeat(W, Z.shape[-1], axis=2)
                Z[:, -1, :, :] = torch.tensor(W)

        elif X.ndim == 2 and Y.ndim == 2:
            Z = torch.einsum("wp,wq->wqp", X, Y)

        else:
            raise ValueError(f"Unhandled tensor dimensions: {X.ndim} (x), {Y.ndim} (y)")

        return Z


    
    def generate_batch(self, size=4):
        """
        Generate a batch of data:

        Input:      - size  Batch size

        Outputs:    - X     Batch of inputs
                    - y     Batch of outputs
        """

        X = []
        y = []

        for i in range(size):
            Xi, _, yi = self.__getitem__(0)
            X.append(Xi.unsqueeze(0))
            y.append(yi.unsqueeze(0))

        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)

        return X, y
    
    
    
    def __len__(self):
        """
        Dummy function for pytorch to work properly
        """

        return int(1e12)
    


    def __getitem__(self, _):
        """
        Generate an input
        """

        # Generate 1D spectra
        X, _, iso_x, ws, ws_enc = self.xgen.__getitem__(0, return_ws=True)
        Y, _, iso_y = self.ygen.__getitem__(0, ws=ws)

        # Make 2D spectra from 1D spectra
        Z = self.make_2d(X, Y, ws_enc)
        iso = self.make_2d(iso_x, iso_y)

        # Rotate 2D spectra
        if np.random.random() < self.prot:
            da = self.rrot[1] - self.rrot[0]
            a0 = self.rrot[0]
            a = a0 + da * np.random.random()
            if ws is not None:
                Z[:, :-1] = torch.tensor(sp.ndimage.rotate(Z[:, :-1], a, axes=(-2, -1), reshape=False))
            else:
                Z = torch.tensor(sp.ndimage.rotate(Z, a, axes=(-2, -1), reshape=False))
            iso = torch.tensor(sp.ndimage.rotate(iso, a, axes=(-2, -1), reshape=False))

        if self.noise > 0.:
            Z[:, :, :-1] += torch.randn_like(Z[:, :, :-1]) * self.noise

        return Z, ws, iso