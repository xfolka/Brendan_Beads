import pathlib
import numpy as np
import pandas as pd

from pyometiff import OMETIFFReader, OMETIFFWriter
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import ball, opening, disk, ellipse
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor


def generate_and_visualize_future(filePath : pathlib.Path):
    future = ThreadPoolExecutor().submit(generate_and_visualize,filePath)
    return future


def generate_and_visualize(filePath : pathlib.Path):
    load_and_generate_data(filePath)
    return get_layers_to_visualize(filePath)


def load_and_generate_data(filePath : pathlib.Path):

    # Load the full z stack, this will be handled later by napari
    reader = OMETIFFReader(fpath=filePath)
    img_array, metadata, xml_metadata = reader.read()

    channel_0 = img_array[0]  # shape: (140, 512, 512)
    channel_1 = img_array[1]  # shape: (140, 512, 512)

    ch1_filtered = gaussian_filter(channel_1, sigma=[2.,1.,1.])
    filt_op = np.max(ch1_filtered, axis=2)
    thresh = threshold_otsu(filt_op)

    # Apply threshold and label
    binary = ch1_filtered > thresh
    blobs = label(binary)
    assert blobs.max()<65000, "issues here"
    blobs = blobs.astype(np.uint16)

    # `blobs` is a labeled 3D array
    props = regionprops(blobs)

    # Extract centroids
    center_coords = np.array([p.centroid for p in props])  # shape: (N, 3), in (z, y, x)

    fname = filePath.name
    fdir = filePath.absolute()
    fname = fname.split(".ome.tiff")[0]
    fname = fname+"_blobs.ome.tiff"
    omePath = filePath.parent.joinpath(fname)
    print(omePath)

    dimension_order = "ZYX"
    metadata_dict = {
        "PhysicalSizeX" : metadata['PhysicalSizeX'],
        "PhysicalSizeXUnit" : "µm",
        "PhysicalSizeY" : metadata['PhysicalSizeY'],
        "PhysicalSizeXUnit" : "µm",
        "PhysicalSizeZ" : metadata['PhysicalSizeZ'],
        "PhysicalSizeZUnit" : "µm",
        
        "Channels" : {
            "filtered" : {
                "Name" : "blobs",
                "SamplesPerPixel": 1,
            },
        }
    }

    writer = OMETIFFWriter(
        fpath=omePath,
        dimension_order=dimension_order,
        array=blobs,
        metadata=metadata_dict,
        explicit_tiffdata=False)

    writer.write()


    def keep_largest_label(label_img):
        props = regionprops(label_img)
        if not props:
            return np.zeros_like(label_img, dtype=label_img.dtype)

        # Find the region with the largest area
        largest_region = max(props, key=lambda r: r.area)
        largest_label = largest_region.label

        # Create a mask with only the largest label
        return (label_img == largest_label).astype(label_img.dtype)


    ch0_filtered = gaussian_filter(channel_0, sigma=[.3,2,2])
    filt_op = np.mean(ch0_filtered, axis=2)
    thresh = threshold_otsu(filt_op)

    # Apply threshold and label
    surface_bw = ch0_filtered > thresh

    selem = ellipse(5,1)
    img = surface_bw[:,:,0]
    opened = opening(img, selem)

    opened_volume = np.empty_like(surface_bw)

    for x in range(surface_bw.shape[2]):
        img = surface_bw[:,:,x]
        opened = opening(img, selem)
        opened_volume[:,:,x] = opened

    for y in range(opened_volume.shape[2]):
        img = opened_volume[:,y,:]
        opened = opening(img, selem)
        opened_volume[:,y,:] = opened

    surface = label(opened_volume)
    assert surface.max()<65000, "issues here"
    surface = surface.astype(np.uint16)

    surface = keep_largest_label(surface)
    fname = filePath.name
    fname = fname.split(".ome.tiff")[0]
    fname = fname+"_surface.ome.tiff"
    omePath = filePath.parent.joinpath(fname)
    print(omePath)

    dimension_order = "ZYX"
    metadata_dict = {
        "PhysicalSizeX" : metadata['PhysicalSizeX'],
        "PhysicalSizeXUnit" : "µm",
        "PhysicalSizeY" : metadata['PhysicalSizeY'],
        "PhysicalSizeXUnit" : "µm",
        "PhysicalSizeZ" : metadata['PhysicalSizeZ'],
        "PhysicalSizeZUnit" : "µm",
        
        "Channels" : {
            "filtered" : {
                "Name" : "surface",
                "SamplesPerPixel": 1,
            },
        }
    }

    writer = OMETIFFWriter(
        fpath=omePath,
        dimension_order=dimension_order,
        array=surface,
        metadata=metadata_dict,
        explicit_tiffdata=False)

    writer.write()


    # Physical voxel size: (z, y, x)
    voxel_size = np.array([metadata['PhysicalSizeZ'], metadata['PhysicalSizeY'], metadata['PhysicalSizeX']])  # in microns

    edges = find_boundaries(surface>0, mode='outer')
    edge_coords = np.column_stack(np.nonzero(edges))

    # Scale coordinates to real physical units
    edge_coords_phys = edge_coords * voxel_size
    center_coords_phys = center_coords * voxel_size

    # Build KD-tree in physical space
    tree = cKDTree(edge_coords_phys)
    dists, indices = tree.query(center_coords_phys, k=1)

    # Find closest edge point for each center
    nearest_edge_coords = edge_coords[indices]  # shape: (n, 3)
    nearest_edge_coords_phys = edge_coords_phys[indices]

    # Compute displacement vectors
    displacements = nearest_edge_coords - center_coords  # shape: (n, 3)
    displacements_phys = nearest_edge_coords_phys - center_coords_phys  # shape: (n, 3)

    # Build table
    vector_table = pd.DataFrame({
        'z0': center_coords[:, 0],
        'y0': center_coords[:, 1],
        'x0': center_coords[:, 2],
        'z1': nearest_edge_coords[:, 0],
        'y1': nearest_edge_coords[:, 1],
        'x1': nearest_edge_coords[:, 2],
        'dz': displacements[:, 0],
        'dy': displacements[:, 1],
        'dx': displacements[:, 2],
        'dz_phys': displacements_phys[:, 0],
        'dy_phys': displacements_phys[:, 1],
        'dx_phys': displacements_phys[:, 2],

        'distance': dists
    })

    # Save to CSV
    fname = filePath.name
    fname = fname.split(".ome.tiff")[0]
    fname = fname + "_vector_table.csv"

    csv_path = filePath.parent.joinpath(fname)
    vector_table.to_csv(csv_path, index=False)
    
    
def get_layers_to_visualize(imgPath : pathlib.Path):
    # Load the full z stack, this will be handled later by napari
    img_name = imgPath.name
    img_name = img_name.split(".ome.tiff")[0]

    reader = OMETIFFReader(fpath=imgPath)
    img_array, metadata, xml_metadata = reader.read()
    nframes = img_array.shape[0]

    channel_0 = img_array[0]  # shape: (140, 512, 512)
    channel_1 = img_array[1]  # shape: (140, 512, 512)

    channel_0.shape

    surface_path = pathlib.Path(f"./data/{img_name}_surface.ome.tiff")
    reader = OMETIFFReader(fpath=surface_path)
    surface, metadata, xml_metadata = reader.read()
    surface.shape

    blobs_path = pathlib.Path(f"./data/{img_name}_blobs.ome.tiff")
    reader = OMETIFFReader(fpath=blobs_path)
    blobs, metadata, xml_metadata = reader.read()

    df = pd.read_csv(f"./data/{img_name}_vector_table.csv")

    # Reconstruct Napari vectors array
    starts = df[['z0', 'y0', 'x0']].to_numpy()
    ends   = df[['z1', 'y1', 'x1']].to_numpy()
    ends = np.ones_like(starts)*30
    shifts = df[['dz', 'dy', 'dx']].to_numpy()
    vectors = np.stack([starts.astype(np.float32), shifts.astype(np.float32)], axis=1)  # shape: (n, 2, 3)

    voxel_size = np.array([metadata['PhysicalSizeZ'], metadata['PhysicalSizeY'], metadata['PhysicalSizeX']])  # in microns

    return channel_0, "image", surface, "surface", blobs, "blobs", vectors,"vectors", voxel_size
