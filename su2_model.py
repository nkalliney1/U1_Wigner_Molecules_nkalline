import tenpy.linalg.np_conserved as npc
from tenpy.models.model import CouplingMPOModel
import numpy as np
import matplotlib.pyplot as plt
from tenpy.models.lattice import SimpleLattice

np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150

import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.models.lattice import get_lattice

tenpy.tools.misc.setup_logging(to_stdout="INFO")

from matplotlib.collections import LineCollection


class ModTriangular(SimpleLattice):
    """Copied from the tenpy but changed basis"""
    dim = 2  #: the dimension of the lattice

    def __init__(self, Lx, Ly, site, **kwargs):
        sqrt3_half = 0.5 * np.sqrt(3)  # = cos(pi/6)
        basis = np.array([[1.0, 0], [0.5,  sqrt3_half]])
        NN = [(0, 0, np.array([1, 0])), (0, 0, np.array([-1, 1])), (0, 0, np.array([0, -1]))]
        nNN = [(0, 0, np.array([2, -1])), (0, 0, np.array([1, 1])), (0, 0, np.array([-1, 2]))]
        nnNN = [(0, 0, np.array([2, 0])), (0, 0, np.array([0, 2])), (0, 0, np.array([-2, 2]))]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs['pairs'].setdefault('next_next_nearest_neighbors', nnNN)
        SimpleLattice.__init__(self, [Lx, Ly], site, **kwargs)

    def plot_coupling_correlations(self, ax, correlations, coupling=None, wrap=False,
                                cmap='viridis', vmin=None, vmax=None, value_func=np.abs,
                                linewidth=3, cbar=True, cbar_kwargs=None, **kwargs):
        """Plot lines connecting nearest neighbors, colored by a correlation matrix.

        Like :meth:`plot_coupling`, but instead of every bond having the same color, the bond
        connecting MPS site `i` to MPS site `j` is colored according to
        ``value_func(correlations[i, j])``, with a colorbar showing the mapping.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes on which we should plot.
        correlations : 2D array
            Correlation matrix as e.g. returned by
            :meth:`~tenpy.networks.mps.MPS.correlation_function`, with ``correlations[i, j]``
            the value for MPS sites `i`, `j`. Should have shape ``(self.N_sites, self.N_sites)``
            (i.e. cover (at least) one MPS unit cell). For an infinite lattice with ``wrap=True``,
            indices are taken mod ``self.N_sites``, which is correct as long as `correlations`
            is translation invariant by one unit cell.
        coupling : list of (u1, u2, dx)
            Same as in :meth:`plot_coupling`; defaults to ``self.pairs['nearest_neighbors']``.
        wrap : bool
            Same as in :meth:`plot_coupling`.
        cmap : str | Colormap
            Colormap mapping correlation values to colors.
        vmin, vmax : float | None
            Color scale limits; default to the data's min/max.
        value_func : callable
            Applied to the (possibly complex) correlation values before coloring,
            e.g. ``np.abs`` (default), ``np.real``, or ``lambda x: x``.
        linewidth : float
            Width of the plotted bonds (overridden by ``kwargs['linewidth']`` if given).
        cbar : bool
            Whether to attach a colorbar to `ax`.
        cbar_kwargs : dict | None
            Extra keyword arguments passed to ``ax.figure.colorbar``.
        **kwargs :
            Further keyword arguments given to :class:`~matplotlib.collections.LineCollection`.

        Returns
        -------
        lc : :class:`matplotlib.collections.LineCollection`
            The collection added to `ax` (e.g. for further tweaking with ``lc.set_array``).
        """
        if coupling is None:
            coupling = self.pairs['nearest_neighbors']
        correlations = np.asarray(correlations)
        Ls = np.array(self.Ls)
        N_sites = self.N_sites

        all_pos1, all_pos2, all_vals = [], [], []

        for u1, u2, dx in coupling:
            if wrap:
                mps_i, mps_j, _, _ = self.possible_couplings(u1, u2, dx)
                pos1 = self.position(self.mps2lat_idx(mps_i))
                pos2 = self.position(self.mps2lat_idx(mps_j))
                idx1 = np.mod(mps_i, N_sites)
                idx2 = np.mod(mps_j, N_sites)
            else:
                dx = np.r_[np.array(dx), u2 - u1]  # append the difference in u to dx
                lat_idx_1 = self.order[self._mps_fix_u[u1], :]
                lat_idx_2 = lat_idx_1 + dx[np.newaxis, :]
                lat_idx_2_mod = np.mod(lat_idx_2[:, :-1], Ls)
                keep = self._keep_possible_couplings(lat_idx_2_mod, lat_idx_2[:, :-1], u2)
                pos1 = self.position(lat_idx_1[keep, :])
                pos2 = self.position(lat_idx_2[keep, :])
                idx1 = self._mps_fix_u[u1][keep]
                lat_idx_2_mod_full = np.concatenate([lat_idx_2_mod, lat_idx_2[:, -1:]], axis=1)
                idx2 = self.lat2mps_idx(lat_idx_2_mod_full[keep])

            all_pos1.append(pos1)
            all_pos2.append(pos2)
            all_vals.append(correlations[idx1, idx2])

        pos1 = np.concatenate(all_pos1, axis=0)
        pos2 = np.concatenate(all_pos2, axis=0)
        vals = value_func(np.concatenate(all_vals, axis=0))

        if pos1.shape[1] == 1:
            pos1 = pos1 * np.array([[1.0, 0]])  # broadcast a zero column for 1D lattices
            pos2 = pos2 * np.array([[1.0, 0]])
        if pos1.shape[1] != 2:
            raise ValueError('can only plot in 2 dimensions.')

        segments = np.stack([pos1, pos2], axis=1)  # shape (N_bonds, 2, 2)

        kwargs.setdefault('linewidth', linewidth)
        lc = LineCollection(segments, cmap=cmap, **kwargs)
        lc.set_array(vals)
        if vmin is not None or vmax is not None:
            lc.set_clim(vmin, vmax)
        ax.add_collection(lc)
        ax.autoscale()

        if cbar:
            cbar_kwargs = dict(cbar_kwargs or {})
            cbar_kwargs.setdefault('ax', ax)
            ax.figure.colorbar(lc, **cbar_kwargs)
        return lc

class SU2Model(CouplingMPOModel):
    #model_params: lat_type; order; Lx; Ly; bc_x; bc_y; bc_MPS; J's

    def init_sites(self, model_params):
        #we are conserving a U(1) quantity (called Sz)
        #basis state 1 (spin up) has "charge" 1, down has -1

        charge_info = npc.ChargeInfo([1], ['Sz'])
        ch = npc.LegCharge.from_qflat(charge_info, [1, 1, -1 ,-1])

        #define local operators manually and label the cols/rows with their charge
        Sz = npc.Array.from_ndarray(np.kron([[0.5, 0.0], [0.0, -0.5]], np.eye(2,2)), [ch, ch.conj()])
        Sp = npc.Array.from_ndarray(np.kron([[0.0, 1.0], [0.0, 0.0]], np.eye(2,2)), [ch, ch.conj()])
        Sm = npc.Array.from_ndarray(np.kron([[0.0, 0.0], [1.0, 0.0]], np.eye(2,2)), [ch, ch.conj()])

        Etaz = npc.Array.from_ndarray(np.kron( np.eye(2,2), [[0.5, 0.0], [0.0, -0.5]]), [ch, ch.conj()])
        Etap = npc.Array.from_ndarray(np.kron( np.eye(2,2), [[0.0, 1.0], [0.0, 0.0]]), [ch, ch.conj()])
        Etam = npc.Array.from_ndarray(np.kron( np.eye(2,2), [[0.0, 0.0], [1.0, 0.0]]), [ch, ch.conj()])

        #create the site
        site = tenpy.networks.site.Site(ch, ["upup", "updown", "downup","downdown"], 
                                        Sz = Sz, Sp = Sp, Sm = Sm, Etaz = Etaz, Etap = Etap, Etam=Etam)
        return site
    
    def init_lattice(self, model_params):
        #get the class of the lattice type
        lat_type = model_params.get("lat_type", "Square")
        if not isinstance(lat_type, str):
            raise TypeError("Lattice type must be a string")
        
        LatticeClass = get_lattice(lattice_name = lat_type)
        if not LatticeClass.dim == 2:
            raise Exception("Lattice must be 2D")
        
        #order = the way we wind the 1D chain around the 2D lattice
        order = model_params.get("order", "Cstyle")

        sites = self.init_sites(model_params)

        #Do iDMRG
        bc_MPS = model_params.get("bc_MPS", "infinite")
        #open, periodic
        bc_x = model_params.get("bc_x", "periodic")
        #default = periodic for infinite DMRG
        bc_y = model_params.get("bc_y", "periodic")
        #how we wind the chain around the 2D lattice
        order = model_params.get("order", "Cstyle")

        #dimensions
        #length of unit cell in x direction
        Lx = model_params.get('Lx', 1)
        #cylinder width
        Ly = model_params.get('Ly', 4)
        
        lat = LatticeClass(Lx, Ly, sites, bc = [bc_x, bc_y], order = order, bc_MPS = bc_MPS)
        return lat
    
    def angle_of_bond(self, u1, u2, dx):
        """Returns the angle in degrees of a bond (u1, u2, dx)."""
        lat = self.lat
        pos1 = lat.unit_cell_positions[u1]
        pos2 = lat.unit_cell_positions[u2]

        # Absolute position difference including unit cell shift
        delta = pos2 - pos1 + dx @ lat.basis  # lat.basis are the lattice vectors

        angle = np.degrees(np.arctan2(delta[1], delta[0]))
        return angle
    
    def init_terms(self, model_params):

        #s = spin term couplings
        #o = orbital-only term couplings
        Jh = model_params.get("Jh", 1.0)
        Jzzo = model_params.get("Jzzo", 1.0)
        Jpmo = model_params.get("Jpmo", 1.0)
        Jpmo_exp = np.exp(1.j*model_params.get("Jpmo_phase",1.0))
        Jppo = model_params.get("Jppo", 1.0)
        Jp = model_params.get("Jp", 1.0)
        Jp_exp = np.exp(1.j*model_params.get("Jp_phase", 1.0))
        Jzzs = model_params.get("Jzzs", 1.0)
        Jpms = model_params.get("Jpms", 1.0)
        Jpms_exp = np.exp(1.j*model_params.get("Jpms_phase",1.0))
        Jpps = model_params.get("Jpps", 1.0)
        #the nu_ij's depending on the bond angle
        phases = {0.0: 1.0, 120.0: np.exp(2*np.pi*1.j / 3.0 ), -120.0: (np.exp(2*np.pi*1.j / 3.0 ))**2}

        #add couplings to hamiltonian
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            #SPIN TERMS
            #Jh(Sz_1*Sz_2 + 1/2 S+ S- + 1/2 S- S+)
            self.add_coupling(Jh, u1, 'Sz', u2, 'Sz', dx, plus_hc = False)
            self.add_coupling(Jh * 0.5, u1, 'Sp', u2, 'Sm', dx, plus_hc = True)
            #Hzz
            self.add_coupling(Jzzs, u1, "Etaz Sz", u2, "Etaz Sz", dx)
            self.add_coupling(Jzzs * 0.5, u1, "Etaz Sp", u2, "Etaz Sm", dx, plus_hc = True)
            #H+-
            self.add_coupling(Jpms_exp*Jpms, u1, "Etap Sz", u2, "Etam Sz", dx, plus_hc = True)
            self.add_coupling(Jpms_exp*Jpms * 0.5, u1, "Etap Sp", u2, "Etam Sm", dx, plus_hc = True)
            self.add_coupling(Jpms_exp*Jpms * 0.5, u1, "Etap Sm", u2, "Etam Sp", dx, plus_hc = True)
            #H++
            angle = self.angle_of_bond(u1, u2, dx)
            self.add_coupling(Jpps * phases[round(angle, 1)], u1, "Etap Sz", u2, "Etap Sz", dx, plus_hc = True)
            self.add_coupling(0.5 * Jpps * phases[round(angle, 1)], u1, "Etap Sp", u2, "Etap Sm", dx, plus_hc = True)
            self.add_coupling(0.5 * Jpps * phases[round(angle, 1)], u1, "Etap Sm", u2, "Etap Sp", dx, plus_hc = True)
            #H+
            self.add_coupling(Jp_exp*Jp*np.conjugate(phases[round(angle, 1)]), u1, "Etap Sz", u2, "Sz", dx, plus_hc = True)
            self.add_coupling(Jp_exp*Jp*phases[round(angle, 1)], u1, "Sz", u2, "Etam Sz", dx, plus_hc = True)
            self.add_coupling(0.5*Jp_exp*Jp*np.conjugate(phases[round(angle, 1)]), u1, "Etap Sp", u2, "Sm", dx, plus_hc = True)
            self.add_coupling(0.5*Jp_exp*Jp*phases[round(angle, 1)], u1, "Sp", u2, "Etam Sm", dx, plus_hc = True)
            self.add_coupling(0.5*Jp_exp*Jp*np.conjugate(phases[round(angle, 1)]), u1, "Etap Sm", u2, "Sp", dx, plus_hc = True)
            self.add_coupling(0.5*Jp_exp*Jp*phases[round(angle, 1)], u1, "Sm", u2, "Etam Sp", dx, plus_hc = True)

            #ORBITAL TERMS
            #Hzz
            self.add_coupling(Jzzo, u1, "Etaz", u2, "Etaz", dx)
            #H+-
            self.add_coupling(Jpmo_exp*Jpmo, u1, "Etap", u2, "Etam", dx, plus_hc = True)
            #H++
            self.add_coupling(Jppo*phases[round(angle, 1)], u1, "Etap", u2, "Etap", dx, plus_hc = True)

def m_corr_function_zz(results, psi, model, simulation, results_key="corr_function_zz"):
    corr=psi.correlation_function('Sz', 'Sz', range(10), range(10))
    results["corr_function_zz"]=corr

def m_corr_function_xx(results, psi, model, simulation, results_key="corr_function_xx"):
    corr_pm = psi.correlation_function('Sp', 'Sm', range(10), range(10))
    corr_mp = psi.correlation_function('Sm', 'Sp', range(10), range(10))
    corr = [[sum(x)/2.0 for x in zip(corr_pm[i], corr_mp[i])] for i in range(len(corr_pm))]
    results["corr_function_xx"] = corr

def corr_function_ss(psi, sites1 = None, sites2 = None):
    corr_zz = psi.correlation_function('Sz', 'Sz', sites1, sites2)
    corr_pm = psi.correlation_function('Sp', 'Sm', sites1, sites2)
    corr_mp = psi.correlation_function('Sm', 'Sp', sites1, sites2)
    #sum up term-by-term
    corr = corr_zz + 0.5 * (corr_pm + corr_mp)
    #should be real so just take the real part
    return np.real(corr)

def corr_function_etap_etam(psi, sites1 = None, sites2 = None):
    corr_pm = psi.correlation_function('Etap', 'Etam', sites1, sites2)
    return corr_pm
