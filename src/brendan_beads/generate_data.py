import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from pyometiff import OMETIFFReader, OMETIFFWriter
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import ellipse, opening
from skimage.segmentation import find_boundaries

DATA_GEN_DIR = "_generated"


def generate_and_visualize_future(blobs_data_layer, surface_data_layer, meta_data, filePath: pathlib.Path):
    future = ThreadPoolExecutor().submit(generate_and_visualize, blobs_data_layer, surface_data_layer, meta_data, filePath)
    return future


def generate_and_visualize(blobs_data_layer, surface_data_layer, metadata, filePath: pathlib.Path):
    generate_data(blobs_data_layer, surface_data_layer, metadata, filePath)
    
    #load_and_generate_data(filePath)
    return get_layers_to_visualize(filePath)


def get_generated_data_dir(baseFile: pathlib.Path) -> pathlib.Path:
    f_name = baseFile.name
    base_name = f_name.split(".ome.tiff")[0]
    gen_dir = baseFile.parent.joinpath(base_name + DATA_GEN_DIR)
    return gen_dir


def get_and_create_data_dir(filePath: pathlib.Path):
    g_dir = get_generated_data_dir(filePath)
    g_dir_path = filePath.parent.joinpath(g_dir)
    g_dir_path.mkdir(exist_ok=True)
    return g_dir_path


def get_metada_pixelsizes(metadata: dict) -> dict:
    metadata_dict = {
        "PhysicalSizeX": metadata['PhysicalSizeX'],
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": metadata['PhysicalSizeY'],
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": metadata['PhysicalSizeZ'],
        "PhysicalSizeZUnit": "µm"
    }
    return metadata_dict


def generate_data(blobs_data_layer, surface_data_layer, metadata, filePath: pathlib.Path):
    
    g_dir_path = get_and_create_data_dir(filePath)
    fname = filePath.name
    
    blobs = generate_blobs_data_for(blobs_data_layer, metadata, fname, g_dir_path)
    surfaces = generate_surfaces_data_for(surface_data_layer, metadata, fname, g_dir_path)
    generate_vectors_data_for(blobs, surfaces, metadata, fname, g_dir_path)


def generate_blobs_data_for(img_data, metadata, file_name, g_dir_path):
    ch1_filtered = gaussian_filter(img_data, sigma=[2., 1., 1.])
    filt_op = np.max(ch1_filtered, axis=2)
    thresh = threshold_otsu(filt_op)

    # Apply threshold and label
    binary = ch1_filtered > thresh
    blobs = label(binary)
    assert blobs.max()<65000, "issues here"
    blobs = blobs.astype(np.uint16)

    fname = file_name.split(".ome.tiff")[0]
    fname = fname+"_blobs.ome.tiff"
    omePath = g_dir_path.joinpath(fname)

    dimension_order = "ZYX"
    metadata_dict = get_metada_pixelsizes(metadata)
    metadata_dict["Channels"] = {
                "filtered": {
                    "Name": "blobs",
                    "SamplesPerPixel": 1
                }
            }

    writer = OMETIFFWriter(
        fpath=omePath,
        dimension_order=dimension_order,
        array=blobs,
        metadata=metadata_dict,
        explicit_tiffdata=False)

    writer.write()

    return blobs


def generate_surfaces_data_for(img_data, metadata, file_name, g_dir_path):

    def keep_largest_label(label_img):
        props = regionprops(label_img)
        if not props:
            return np.zeros_like(label_img, dtype=label_img.dtype)

        # Find the region with the largest area
        largest_region = max(props, key=lambda r: r.area)
        largest_label = largest_region.label

        # Create a mask with only the largest label
        return (label_img == largest_label).astype(label_img.dtype)

    ch0_filtered = gaussian_filter(img_data, sigma=[.3, 2, 2])
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
    fname = file_name.split(".ome.tiff")[0]
    fname = fname+"_surface.ome.tiff"
    omePath = g_dir_path.joinpath(fname)

    dimension_order = "ZYX"
    metadata_dict = get_metada_pixelsizes(metadata)
    metadata_dict["Channels"] = {
                "filtered": {
                    "Name": "surface",
                    "SamplesPerPixel": 1
                }
            }

    writer = OMETIFFWriter(
        fpath=omePath,
        dimension_order=dimension_order,
        array=surface,
        metadata=metadata_dict,
        explicit_tiffdata=False)

    writer.write()
    
    return surface


def generate_vectors_data_for(blobs_img_data, surface_img_data, metadata, file_name, g_dir_path):
    
    # `blobs` is a labeled 3D array
    props = regionprops(blobs_img_data)

    # Extract centroids
    center_coords = np.array([p.centroid for p in props])  # shape: (N, 3), in (z, y, x)
    
    # Physical voxel size: (z, y, x)
    voxel_size = np.array([metadata['PhysicalSizeZ'], metadata['PhysicalSizeY'], metadata['PhysicalSizeX']])  # in microns

    edges = find_boundaries(surface_img_data>0, mode='outer')
    edge_coords = np.column_stack(np.nonzero(edges))

    # Scale coordinates to real physical units
    edge_coords_phys = edge_coords * voxel_size
    center_coords_phys = center_coords * voxel_size

    # Build KD-tree in physical space
    tree = KDTree(edge_coords_phys)
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

    #Save to CSV
    fname = file_name.split(".ome.tiff")[0]
    fname = fname + "_vector_table.csv"

    csv_path = g_dir_path.joinpath(fname)
    vector_table.to_csv(csv_path, index=False)


def get_layers_to_visualize(imgPath : pathlib.Path):
    # Load the full z stack, this will be handled later by napari
    img_name = imgPath.name
    img_name = img_name.split(".ome.tiff")[0]

    reader = OMETIFFReader(fpath=imgPath)
    img_array, metadata, xml_metadata = reader.read()

    channel_0 = img_array[0]  # shape: (140, 512, 512)

    #get data generate dir
    dataGenDir = get_generated_data_dir(imgPath)
    surface_path = dataGenDir.joinpath(f"{img_name}_surface.ome.tiff")

    reader = OMETIFFReader(fpath=surface_path)
    surface, metadata, xml_metadata = reader.read()

    blobs_path = dataGenDir.joinpath(f"{img_name}_blobs.ome.tiff")
    reader = OMETIFFReader(fpath=blobs_path)
    blobs, metadata, xml_metadata = reader.read()

    vectors_path = dataGenDir.joinpath(f"{img_name}_vector_table.csv")
    df = pd.read_csv(vectors_path)

    # Reconstruct Napari vectors array
    starts = df[['z0', 'y0', 'x0']].to_numpy()
    ends   = df[['z1', 'y1', 'x1']].to_numpy()
    ends = np.ones_like(starts)*30
    shifts = df[['dz', 'dy', 'dx']].to_numpy()
    vectors = np.stack([starts.astype(np.float32), shifts.astype(np.float32)], axis=1)  # shape: (n, 2, 3)

    voxel_size = np.array([metadata['PhysicalSizeZ'], metadata['PhysicalSizeY'], metadata['PhysicalSizeX']])  # in microns

    return channel_0, "image", surface, "surface", blobs, "blobs", vectors, "vectors", voxel_size


def read_ome_tiff(file: pathlib.Path):
    reader = OMETIFFReader(fpath=file)
    image, metadata, xml_metadata = reader.read()
    return image, metadata
