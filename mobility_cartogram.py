from collections import defaultdict
import pandas as pd
import geopandas as gpd
import imageio
import numpy as np
from subprocess import call,Popen,PIPE
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from shapely.affinity import affine_transform
from shapely.geometry import Polygon,Point,MultiPoint
from sys import argv
from geocube.api.core import make_geocube
import matplotlib.pyplot as plt
import os
from pycartogram import Cartogram

if not "DISPLAY" in os.environ or not os.environ["DISPLAY"]:
    plt.switch_backend('agg')

def parse_mobility_csv(fname):
    output = defaultdict(lambda : defaultdict(int))
    for row in open(fname):
        x = row.strip().split(",")
        output[x[-1]][x[1]]+=1
    return output

def df_from_dict(mobdict):
    output=[]
    for k in mobdict:
        for k2 in mobdict[k]:
            output.append((k,k2,mobdict[k][k2]))
    df = pd.DataFrame(output)
    df.columns = ["bg","timestamp","devices"]
    return df

def parse_mobility_json(fname):
    mobdict = json.load(open(fname))
    return df_from_dict(mobdict)

def log_step(stepname):
    #todo : add logging timing etc.
    print(stepname)
    
def load_polygon_df(fname):
    polygon_df = gpd.read_file(fname).to_crs(projection)
    
def custom_join(polygon_df,data_df,data_key="devices"):
    df_devices = polygon_df.join(data_df,how="left")
    df_devices[data_key] = df_devices[data_key] / df_devices.area
    mean_val = df_devices[df_devices.ALAND > 0][data_key].mean()
    df_devices.loc[(df_devices.ALAND <= 0, data_key)] = mean_val
    df_devices.loc[(df_devices[data_key].isna(), data_key)] = 0
    return df_devices, mean_val
    



if __name__ == "__main__":
    mobility_data_fname = "/home/micha/AggData/20200310.csv" if len(argv) < 2 else argv[1]
    polygon_data_fname = "/home/micha/devsnaps/cart-1.2.2/nyc_bg.geojson"
    polygons_to_scale_fname = "/home/micha/nyc_census_tracts.geojson"
    cartogram_polygon_fname = "nyc_tag.geojson" 
    freq = "1H" 
    external_mean_val = None if len(argv) < 3 else float(argv[2])
    external_mean_val_weight = 0 if len(argv) < 4 else float(argv[3])
    dfad = pd.read_csv(mobility_data_fname,dtype={"bg":str,"id_count":float},parse_dates=["time"])
    dfadp = dfad.pivot("time","bg")
    dfadp = dfadp.resample(freq).interpolate().fillna(0)
    for t,d in dfadp.iterrows():
        mc = Cartogram()
        mc.imsize = (800,1000)
        mc.pad_width = (300,300)
        timestring = t.isoformat()
        cartogram_polygon_fname = f"mobility_cartogram_{timestring}_{freq}-{external_mean_val}-{external_mean_val_weight}.geojson"
        data_df = pd.DataFrame(d.droplevel(level=0)).rename(columns=lambda x: mc.data_key)
        log_step("Loading data")
        mc.load_polygon_df(polygon_data_fname,"GEOID")
        mc.polygon_df = mc.polygon_df[mc.polygon_df.COUNTYFP == "061"]
        mc.polygon_df,mc.mean_val = custom_join(mc.polygon_df,data_df)
        if external_mean_val is not None:
            print("Found external_mean_val %.12f and weight %.2f"%(external_mean_val,external_mean_val_weight))
            original_mean_val = mc.mean_val
            mc.mean_val = external_mean_val*external_mean_val_weight + mc.mean_val*(1 - external_mean_val_weight)
            mc.polygon_df.loc[(mc.polygon_df.ALAND <= 0, mc.data_key)] = mc.mean_val
            print("Using mean val %.12f instead of %.12f"%(mc.mean_val,original_mean_val))
        log_step("Preparing data for cart")
        mc.create_geocube_density_map()
        log_step("Running cart")
        mc.run_cart_on_density_map()
        log_step("Loading cart results")
        mc.load_interpolated_output()
        plt.figure(figsize=tuple(reversed(np.round(16*np.array(mc.padded_im.shape).T/max(mc.padded_im.shape)))))
        mc.plot_density_map()
        plt.title(timestring)
        plt.tight_layout()
        plt.axis("equal")
        plt.savefig(cartogram_polygon_fname.rstrip(".geojson")+".png")
        plt.close()
        mc.calculate_grid_transforms()
        log_step("Transforming polygons")
        original_df = mc.polygon_df.copy()
        mc.polygon_df = gpd.read_file(polygons_to_scale_fname).to_crs(mc.projection)
        mc.polygon_df = mc.polygon_df[mc.polygon_df["boro_name"] == "Manhattan"]
        mc.get_cartogram_transformed_data()
        log_step("Saving new file")
        dftag =  mc.polygon_df.drop(columns=["old geometry"]).to_crs("EPSG:4326")
        call(["rm", cartogram_polygon_fname])
        dftag.to_file(cartogram_polygon_fname,driver="GeoJSON")
        
