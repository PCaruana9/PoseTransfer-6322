#import trimesh
import numpy as np
import MyMesh as MM
import meshio


#Evaluates the Pointwise Mesh euclidean Distance specified in original Paper

def evaluate(out_mesh, gt_mesh, debug=False):
    out_verts= out_mesh.points
    out_bbox= MM.bbox(out_mesh)

    gt_verts = gt_mesh.points
    gt_bbox = MM.bbox(gt_mesh)

    # here we are centering each mesh around (0,0), by taking the lazy centroid via the
    # center of the mesh's bounding box, and subtracting that for each point. Frankly I think
    # a true center of mass would be better but oh well. This is what the original authors did.
    # This could be simply to avoid issues regarding mesh density, hard to say.
    out_verts_centered = out_verts - (out_bbox[0] + out_bbox[1]) / 2
    gt_verts_centered = gt_verts - (gt_bbox[0] + gt_bbox[1]) / 2

    PMD = (np.mean((out_verts_centered-gt_verts_centered)**2))
    if debug: print("PMD = " + str(PMD))
    return PMD


