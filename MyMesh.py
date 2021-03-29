
import meshio
import numpy as np

#Author: Peter Caruana
#Email:  caruana9@my.yorku.ca

# This function library exists as a wraper for meshio to implement PyMesh features
# Why? Because pip install PyMesh doesnt give you the full version of PyMesh, and
# god help you if you are trying to build and install PyMesh2 on google colab

#takes meshio mesh as input
#implements PyMesh bbox feature
#returns bounding vertex coordinates of mesh
#[0] are minimum values, [1] are maximum values
def bbox(mesh):
  columns = list(zip(*mesh.points))
  max = [np.max(row) for row in columns]
  min = [np.min(row) for row in columns]
  
  return np.array([min, max])

def getFaces(mesh):
  return mesh.cells[0][1]

def saveMesh(fname, points, faces):
  cells = [("triangle", faces)]
  meshio.Mesh(points, cells).write(fname)