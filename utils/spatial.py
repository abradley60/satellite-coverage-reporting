import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.mask
import numpy as np
import cv2
import matplotlib.pyplot as plt
import geopandas as gpd
from PIL import Image, ImageFont, ImageDraw 
import imageio
from shapely.geometry import Polygon, Point

def find_file(folder, endswith):
    for root, dirs, files in os.walk(folder):
        for name in files:
            if name.endswith(endswith):
                filename = os.path.join(root,name)
                return filename
            
def bbox_around_point(x,y,h):
    aoi_point = Point(y, x)
    # Define the four points as (x, y) coordinates around Davis station in 3031
    point1 = (x-h, y+h)
    point2 = (x+h, y+h)
    point3 = (x+h, y-h)
    point4 = (x-h, y-h)

    # Create a Shapely polygon from the points
    aoi = Polygon([point1, point2, point3, point4])

    return aoi

def reproject_tif(src_path, dst_path, dst_crs):
    
    with rasterio.open(src_path) as src:
        print(f'reprojecting from {src.crs} to {dst_crs}')
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        print(f'saving - {dst_path}')
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def make_bgr_tif(product, crs, data_dir='data', satellite='landsat'):

    #2,3,4 = blue, green, red
    if 'LANDSAT' in satellite.upper():
        path = f'{data_dir}/{product}/{product}'
        band2=rasterio.open(f'{path}_B2.TIF')
        band3=rasterio.open(f'{path}_B3.TIF')
        band4=rasterio.open(f'{path}_B4.TIF')
    if 'SENTINEL' in satellite.upper():
        folder = f'{data_dir}/{product}/{product}.SAFE/GRANULE/'
        band2=rasterio.open(find_file(folder, 'B02_10m.jp2'))
        band3=rasterio.open(find_file(folder, 'B03_10m.jp2'))
        band4=rasterio.open(find_file(folder, 'B04_10m.jp2'))
    band2_geo = band2.profile
    band2_geo.update({"count": 3})

    print(f'making {data_dir}/{product}/bgr.tiff')
    with rasterio.open(f'{data_dir}/{product}/bgr.tiff', 'w', **band2_geo) as dest:
        dest.write(band2.read(1),1)
        dest.write(band3.read(1),2)
        dest.write(band4.read(1),3)

    #reproject into 3031
    dst_crs = f'EPSG:{crs}'
    src_path = f'{data_dir}/{product}/bgr.tiff'
    dst_path = f'{data_dir}/{product}/bgr_{crs}.tif'

    reproject_tif(src_path, dst_path, dst_crs)            
    os.remove(src_path)
    with rasterio.open(dst_path) as dst:
        return dst.read(), dst.meta

def read_cloudmask(product, crs, data_dir='data', satellite='landsat'):
    #2,3,4 = blue, green, red
    if 'LANDSAT' in satellite.upper():
        src_path = f'{data_dir}/{product}/{product}_QA_PIXEL.TIF'
    if 'SENTINEL' in satellite.upper():
        folder = f'{data_dir}/{product}/{product}.SAFE/GRANULE/'
        src_path = rasterio.open(find_file(folder, 'SCL_20m.jp2'))

    #reproject into 3031
    dst_crs = f'EPSG:{crs}'
    dst_path = f'{data_dir}/{product}/CLM_{crs}.tif'
    print(f'saving to {dst_path}')

    reproject_tif(src_path, dst_path, dst_crs)            
    with rasterio.open(dst_path) as dst:
        return dst.read(), dst.meta

def normalise_bands(image, n_bands, p_min=5, p_max=95):
    norm = []
    for c in range(0,n_bands):
        band = image[c,:, :].copy()
        plow, phigh = np.percentile(band, (p_min,p_max))
        band = (band - plow) / (phigh - plow)
        band[band<0] = 0
        band[band>1] = 1
        norm.append(band)
    return np.array(norm) # c,h,w in blue, green, red    

def crop_tif(src_path, dst_path, geometry):
    print(f'cropping - {src_path}')
    with rasterio.open(f'{src_path}') as src:
        out_image, out_transform = rasterio.mask.mask(src, [geometry], crop=True)
        meta = src.meta
        print(out_image.shape)
    return out_image, meta

def crop_bgr_tif(product, crs, geometry, save_dir, data_dir='data',show=True):

    print(f'cropping and normalising - {data_dir}/{product}/bgr_{crs}.tif')
    with rasterio.open(f'{data_dir}/{product}/bgr_{crs}.tif') as src:
        # image is in b,g,r 
        out_image, out_transform = rasterio.mask.mask(src, [geometry], crop=True)
        print(out_image.shape)
        norm = normalise_bands(out_image, n_bands=3, p_min=5, p_max=95)
        # change to hwc and scale to 0-255
        norm_hwc = np.transpose((norm*255).astype(np.uint8), (1, 2, 0))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f'saving - {save_dir}/{product}.png')
        cv2.imwrite(f'{save_dir}/{product}.png', norm_hwc)
        # matplot wants to plot in RBG, hwc
        if show:
            plt.imshow(cv2.cvtColor(norm_hwc, cv2.COLOR_BGR2RGB))
            plt.show()

    return norm_hwc, out_transform

def make_gif_from_images(image_paths, save_path, duration=0.5, scale=1, text='', text_size=16):
    
    # Create a list to store PIL image objects
    images = []

    # Load each image and append it to the list
    for i,image_path in enumerate(image_paths):
        image = Image.open(image_path)

        #scale the image
        width, height = image.size
        dim = (int(width*scale), int(height*scale))   
        # resize image
        image = image.resize(dim)

        # add text to image
        if text:
            draw = ImageDraw.Draw(image)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype('/Library/Fonts/Arial.ttf', text_size)
            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((20, 20),text[i],(255,0,0), font=font)
        images.append(image)

    # Save the list of images as a GIF
    #imageio.mimsave(save_path, images, duration=duration)  # Adjust the duration as needed (in seconds)
    images[0].save(save_path, save_all=True, append_images=images[1:], duration=duration, loop=0)  # Adjust duration as needed (in milliseconds)
    print(f'GIF saved to {save_path}')