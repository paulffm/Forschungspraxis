import gmsh
import sys

gmsh.initialize()
gmsh.model.add("wire")
l_z = 300e-3
r1 = 2e-3
r2 = 3.5e-3

# Inner and outer cable cross-section
circ1 = gmsh.model.occ.add_circle(0, 0, 0, r1)
circ2 = gmsh.model.occ.add_circle(0, 0, 0, r2)
loop1 = gmsh.model.occ.add_curve_loop([circ1])
loop2 = gmsh.model.occ.add_curve_loop([circ2])

    # Extrude to create volume
    # gm.extrude([(1, c1)], 0, 0, -l_z / 20)
    # gm.extrude([(1, c2)], 0, 0, -l_z / 20)

    # Create plane surfaces to connect loops
surf1 = gmsh.model.occ.add_plane_surface([loop1])
surf2 = gmsh.model.occ.add_plane_surface([loop2, loop1])

    # Create physical groups
gmsh.model.occ.synchronize()
wire: int = gmsh.model.add_physical_group(2, [surf1], name="WIRE")
shell: int = gmsh.model.add_physical_group(2, [surf2], name="SHELL")
gnd: int = gmsh.model.add_physical_group(1, [loop2], name="GND")
gmsh.model.mesh.generate(2)
gmsh.write("wire.msh")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
