import matplotlib.pyplot as plt
import cartopy
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.ticker as ticker
import rasterio
import numpy as np
import math
from rasterio.features import rasterize
from tqdm import tqdm
import matplotlib.patches as mpatches
from pyproj import Transformer
import datetime
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import json
import antimeridian
from landsatxplore.earthexplorer import EarthExplorer
from sentinelsat import SentinelAPI

ccrs = cartopy.crs

def plot_results_footprint_map(df, 
                               title='', 
                               group='', 
                               group_colors = {},
                               sort='', 
                               legend_title='', 
                               crs=3031,
                               bounds=(-180, 180, -90, -50)):
    # plot the the product geometries on a map
    east, west, south, north = bounds
    plt.rcParams["figure.figsize"] = [10,8]
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent((east, west, south, north+1), ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    proj = ccrs.SouthPolarStereo() if crs==3031 else ccrs.PlateCarree()
    if not group:
        ax.add_geometries(df.geometry, crs=proj, alpha=0.3, edgecolor='black')
    else:
        groups = df[group].unique() if not sort else df.sort_values(sort)[group].unique()
        # generate N visually distinct colours
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        legend = []
        for i,g in enumerate(groups):
           colour = colors[i] if not group_colors else group_colors[g]
           ax.add_geometries(df[df[group]==g].geometry, crs=proj, alpha=0.3, color=colour) 
           legend.append(mpatches.Patch(color=colour, label=g))
        ax.legend(handles=legend, title= group if not legend_title else legend_title)
    ax.gridlines(draw_labels=True)
    ax.add_feature(cartopy.feature.COASTLINE)
    plt.title(title)
    plt.show()

def plot_timeseries_products(df, title='',stack_col='sat_id', date_col='beginposition',count_freq='7D', plot_freq='1M'):

    sns.set_theme()
    sns.set(rc={'figure.figsize':(10,2)})
    df['round_time'] = df[date_col].dt.round(count_freq)
    count_col = list(df)[0]
    c = df[['round_time',stack_col,count_col]].groupby(['round_time',stack_col]).count().reset_index()
    c = c.pivot(index='round_time', columns=stack_col, values=count_col).fillna(0)
    c = c.resample(plot_freq).max()
    ax = c.plot(kind='bar', stacked=True, width=1)
    ticklabels = ['']
    for i in range(1,len(c.index)):
        ticklabels.append('') if c.index[i].year == c.index[i-1].year else ticklabels.append(c.index[i].year)
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
    ax.set_xlabel('')
    plt.xticks(rotation = 45, fontsize='10')
    plt.yticks(fontsize='10')
    plt.legend(title='')
    plt.title(title)
    set(ticklabels)
    sns.reset_orig()
    plt.show()

def get_freq_and_cloud_raster(df, shape=(1000,1000), cloud_col='cloudcoverpercentage'):
    # convert each polygon of the of the datasets into a raster image (1's for data, 0's for non-data)
    # and progressively add them all up, so the places with 1's data will accumulate into a frequency count
    # replace the zeroes with nans to mask them out from the plot
    crs = ccrs.SouthPolarStereo()
    bounds = crs.boundary.bounds
    transform = rasterio.transform.from_bounds(*bounds, *shape)
    
    freq_raster = np.zeros(shape)
    cc_raster = np.zeros(shape) # approximate cloud cover perc for sentinel 2

    print('making frequency raster')
    for i in tqdm(range(0,len(df))):
        polygon = df['geometry'].iloc[i]
        freq_raster += rasterize([(polygon, 1)], out_shape=shape, transform=transform)
        if cloud_col in list(df):
            cc = df[cloud_col].iloc[i]
            cc_raster += rasterize([(polygon, cc)], out_shape=shape, transform=transform)

    #average the cc percentage
    cc_raster = cc_raster/freq_raster
    #mask out where nodata
    cc_raster[freq_raster==0] = np.nan
    freq_raster[freq_raster==0] = np.nan
    return freq_raster, cc_raster

def plot_frequency(df, title='', cbar_label='Frequency', plot_cloud=False, shape=(1000,1000), cloud_col='cloudcoverpercentage', force_max=False):

    if plot_cloud and (cloud_col not in list(df)):
        print(f'{cloud_col} not in data - set correct cloud cover value to plot')
    freq_raster, cc_raster = get_freq_and_cloud_raster(df, shape=shape, cloud_col=cloud_col)
    raster = cc_raster if plot_cloud else freq_raster
    if force_max:
        raster[raster>force_max] = force_max

    crs = ccrs.SouthPolarStereo()
    bounds = crs.boundary.bounds
    east, west, south, north = -180, 180, -90, -50
    
    plt.rcParams["figure.figsize"] = [10,8]
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent((east, west, south, north+1), ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    color = ax.imshow(raster, origin="upper", extent=(bounds[0], bounds[2], bounds[1], bounds[3]), transform=ccrs.SouthPolarStereo())
    ax.add_feature(cartopy.feature.COASTLINE)
    cbar_max = int(raster[raster>-1].max())
    plt.colorbar(color, ticks=np.linspace(0, cbar_max, 10, dtype=int), label=cbar_label)
    ax.gridlines(draw_labels=True)
    plt.title(title)
    plt.show()
    return ax

def plot_multiple_frequency(df, group, sort_group='', title='', n_cols=2, plot_cloud=False, cbar_label='Pass Frequency', shape=(1000,1000), cloud_col='cloudcoverpercentage', force_max=False):
    
    sort_group = group if not sort_group else sort_group
    df = df.sort_values(sort_group)

    # calculate the size of the figure
    n = df[group].nunique()
    n_rows = math.ceil(n/n_cols)

    # using the variable axs for multiple Axes
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols,
                        subplot_kw={'projection': ccrs.SouthPolarStereo()},
                        figsize=(n_cols*6,n_rows*5.5))
    
    # ensure cloud percentage value exists if to be plotted
    if plot_cloud and (cloud_col not in list(df)):
        print(f'{cloud_col} not in data - set correct cloud cover value to plot')

    # first create the plot raster for each category
    # we can therefore get color to scale for all figures
    raster_dict = {}
    max_freq = 0
    min_freq = 0
    for cat in df[group].unique():
        print(cat)
        cat_data = df[df[group]==cat]
        # get the frequency raster for each cat
        freq_raster, cc_raster = get_freq_and_cloud_raster(cat_data, shape=shape, cloud_col=cloud_col)
        if not plot_cloud:
            raster = freq_raster
        else:
            raster = cc_raster
        
        if force_max:
            raster[raster>force_max] = force_max
        
        r_max = int(raster[raster>-1].max()) if math.isnan(raster.max()) else 1# max
        max_freq = r_max if (r_max > max_freq) else max_freq
        r_min = int(raster[raster>-1].min() ) if math.isnan(raster.min()) else 1
        min_freq = r_min if (r_min < min_freq) else min_freq
        raster_dict[cat] = raster
    
    crs = ccrs.SouthPolarStereo()
    bounds = crs.boundary.bounds
    east, west, south, north = -180, 180, -90, -50
    
    # iterate through the product dict
    count = 0
    print(df[group].unique())
    for cat in df[group].unique():
        print(cat)
        r = math.floor((count)/n_cols)
        c = count % n_cols
        ax_i = (r,c) if ((n_cols > 1) and (n_rows>1)) else count # only a single index needed if no cols
        # get product data for each catagory
        n_products = len(df[df[group]==cat])
        raster = raster_dict[cat]
        ax[ax_i].set_extent((east, west, south, north), ccrs.PlateCarree())
        ax[ax_i].add_feature(cartopy.feature.LAND)
        ax[ax_i].add_feature(cartopy.feature.OCEAN)
        ax[ax_i].title.set_text(f'{cat} ({n_products:,} products)')
        color = ax[ax_i].imshow(raster, 
                                origin="upper", 
                                extent=(bounds[0], bounds[2], bounds[1], bounds[3]), 
                                transform=ccrs.SouthPolarStereo(),
                                vmin=min_freq,
                                vmax=max_freq
                                )
        gl = ax[ax_i].gridlines(draw_labels=True)
        ax[ax_i].add_feature(cartopy.feature.COASTLINE)
        gl.xlabel_style['rotation']= 0
        gl.xlabel_style['ha']= 'center'
        gl.xlabel_style['va']= 'center'
        count += 1

    # add the colorbar for all plots
    plt.tight_layout()
    cbar_ax = fig.add_axes([0.05, -0.03, 0.9, 0.02])
    fig.colorbar(color, cax=cbar_ax, label=cbar_label, orientation="horizontal")
    plt.suptitle(title, y=1.03, fontsize='x-large')

    #delete subplot if uneven number
    if count != (n_rows*n_cols):
        if n_cols > 1:
            fig.delaxes(ax[n_rows-1,n_cols-1])
        else:
            fig.delaxes(ax[n])

    plt.show()

def filter_results_with_geojson(df, filename, plot=False, crs=3031):

    gdf_inclusion = gpd.read_file(filename).set_crs(4326)
    l1 = len(df)
    df = df.to_crs(4326) # convert back to lat lon for intersection
    df = df[df['geometry'].apply(lambda x : x.intersects(gdf_inclusion.geometry.values[0]))]
    print(f'{l1 - len(df)} products have been removed')

    if plot:
        plt.rcParams["figure.figsize"] = [10,8]
        ax = plt.axes(projection=cartopy.crs.PlateCarree(), title='Product Search Area - 50deg south excl. South America and Falkland Islands')
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_geometries(gdf_inclusion.geometry, crs=cartopy.crs.PlateCarree(), alpha=0.7)

    return df.to_crs(crs)

def points_from_poly(poly_list, crs_transformer=False, lon_first=False):
    """Creates a set of lat lons in the crs specified in the native crs or
    crs specified in the crs_transformer.

    Args:
        poly_list (_type_): a list of string polygon points [['-74 175, '-72 170, ...],['-62 ...']]
        crs_transformer (pyproj.Transformer.from_crs, optional): Pyproj transformer to change crs
        lon_first (bool, optional): _description_. True if the lon is first in the points list

    Returns:
        : array of polygon points in target crs
            [[(lat,lon),(lat,lon),(laty,lon)],[(lat,lon),(lat,lon),(laty,lon)]...]
    """
    lon_lat_list = [[float(c) for c in poly[0].split(' ')] for poly in poly_list]
    poly_points = [np.array(l_l).reshape(int(len(l_l)/2),2) for l_l in lon_lat_list]
    if lon_first:
        poly_points = [[list(reversed(x)) for x in lon_lat_list] for lon_lat_list in poly_points]
    if crs_transformer:
        poly_points = [[crs_transformer.transform(p[0],p[1]) for p in  lon_lat_list] for lon_lat_list in poly_points]
    return poly_points

def preprocess_cmr_df(df, crs=3031, lon_first=False, aoi_filter=False, time_start = 'time_start'):
    # preporcess data from cmr api
    print('Converting native polygon points')
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{crs}", always_xy=lon_first)
    transformer = False if crs == 4326 else transformer # set to false if no transformation will be done
    # convert the specified coordinates set to lat, lons in target crs to create polygons
    df['polygons_points'] = df['polygons'].apply(lambda x : points_from_poly(x, transformer, lon_first=lon_first))
    # create polygons from the list of points
    print('Making polygons')
    df['polygons_new'] = df['polygons_points'].apply(lambda x : [Polygon(p) for p in x])
    df['len_polygons'] = df['polygons_new'].apply(lambda x : len(x))
    # create a multpolygon of points or simple polygon 
    print('Making multipolygon')
    df['geometry'] = df['polygons_new'].apply(lambda x : shapely.geometry.MultiPolygon(x)) # if len(x) > 1 else x[0])
    # fix broken polygons
    df['geometry'] = df['geometry'].apply(lambda x : x.buffer(0))
    df = gpd.GeoDataFrame(df, geometry='geometry', crs=f"EPSG:{crs}")

    if aoi_filter:
        print('Filtering to 50deg south excl. South America and Falkland Islands')
        roi_shape = 'shapefiles/50south_excl_argentina_falkand_mid.geojson'
        df = filter_results_with_geojson(df, roi_shape, crs=crs)
        
    # time
    print('Converting times')
    df = df.rename(columns={time_start:'time_start'})
    df['time_start'] = pd.to_datetime(df['time_start'], format="%Y-%m-%dT%H:%M:%S.%fZ")
    df['time_end'] = pd.to_datetime(df['time_end'], format="%Y-%m-%dT%H:%M:%S.%fZ")
    df['month'] = df['time_start'].dt.month
    df['month_name'] = df['time_start'].dt.month_name()
    df['year'] = df['time_start'].dt.year
    MAX_DATE = datetime.datetime.strptime('16/06/23', '%d/%m/%y')
    df = df[df['time_start']<MAX_DATE] #filter for date
    
    #file size
    print('Calculating size')
    if 'granule_size' in df.columns:
        df['size_MB'] = df['granule_size'].astype(float)
        df['size_GB'] = df['size_MB'] / 1_000
        df['size_TB'] = df['size_MB'] / 1_000_000

    return df   