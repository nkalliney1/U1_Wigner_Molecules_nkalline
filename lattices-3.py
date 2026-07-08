import matplotlib.pyplot as plt
from tenpy.models import lattice

Lx, Ly = 4, 3
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(7, 4))

for i, shift in enumerate([1, 0, -1]):
    ax1, ax2 = axes[:, i]
    lat = lattice.Triangular(Lx, Ly, None, bc=['periodic', shift], bc_MPS='infinite')
    for ax in ax1, ax2:
        lat.plot_sites(ax)
        ax.set_aspect('equal')
        ax.set_ylim(-1, 4)
    lat.plot_coupling(ax1)
    lat.plot_bc_identified(ax1, cylinder_axis=True)
    lat.plot_coupling(ax2, wrap=True)
    ax1.set_title('shift = ' + str(shift))
    ax.set_xlim(-1.5)

plt.show()