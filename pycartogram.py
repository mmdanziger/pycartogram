from collections import defaultdict
import pandas as pd
import geopandas as gpd
import imageio
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call,Popen,PIPE
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from shapely.affinity import affine_transform
from shapely.geometry import Polygon,Point,MultiPoint,MultiPolygon
from sys import argv
from geocube.api.core import make_geocube
import tempfile
import atexit

def log_step(stepname):
    #todo : add logging timing etc.
    print(stepname)

class Cartogram(object):
    def __init__(self):
        self.data_df = None
        self.pad_width = (200,200)
        self.imsize = (500,600)
        self.path_to_cart = "/home/micha/devsnaps/cart-1.2.2/cart"
        self.gaussian_blur = None
        self.projection = "EPSG:6539"#"EPSG:4326"
        self.data_key = "devices"
        self.no_data_to_ocean = False
        self.density_map_fname = "/tmp/nyc.dat"
        self.transformed_density_fname = "/tmp/nyc_out.dat"
    
    def generate_temp_files(self):
        self.density_map = tempfile.NamedTemporaryFile()
        self.transformed_density = tempfile.NamedTemporaryFile()
        self.density_map_fname = self.density_map.name 
        self.transformed_density_fname = self.transformed_density.name

    def delete_temp_files(self):
        self.density_map.close()
        self.transformed_density.close()

    def load_polygon_df(self,fname,join_index):
        self.polygon_df = gpd.read_file(fname).to_crs(self.projection).set_index(join_index)
    
    def load_data_df(self,fname,join_index):
        self.data_df = pd.read_csv(fname).set_index(join_index)
    
    def join_polygon_and_data(self):
        df_joined = self.polygon_df.join(dfmob_gb,how="left")
        df_joined[self.data_key] = df_joined[self.data_key] / df_joined.area
        self.mean_val = df_joined[~df_joined[self.data_key].isna()][self.data_key].mean()
        if self.no_data_to_ocean:
            df_joined.loc[(df_joined[self.data_key].isna(), self.data_key)] = self.mean_val
        else:
            df_joined.loc[(df_joined[self.data_key].isna(), self.data_key)] = 0

        self.polygon_df = df_joined
        

    def create_geocube_density_map(self,like=None):
        if like is None:
            minx,miny,maxx,maxy=self.polygon_df.total_bounds
            r = ((maxx - minx) / self.imsize[0])
            print(r)
            self.geocube_resolution = (r,r)
            self.gc = make_geocube(self.polygon_df, [self.data_key],resolution=self.geocube_resolution)
        else:
            self.gc = make_geocube(self.polygon_df, [self.data_key],like=like)
        im = self.gc.to_array().data[0]
        im = np.nan_to_num(im,nan=self.mean_val)
        self.imsize = im.shape
        self.padded_im = np.pad(im,pad_width=self.pad_width,constant_values=self.mean_val)
        if self.gaussian_blur is not None:
            self.padded_im = gaussian_filter(self.padded_im,self.gaussian_blur)
        np.savetxt(self.density_map_fname,self.padded_im)


        
    def run_cart_on_density_map(self):
        cart_command = [self.path_to_cart, str(self.padded_im.shape[1]), str(self.padded_im.shape[0]), self.density_map_fname, self.transformed_density_fname]
        process = call(cart_command)
        return 
        
    def load_interpolated_output(self):
        cart_out = np.loadtxt(self.transformed_density_fname)
        load_shape = (self.padded_im.shape[0]+1,self.padded_im.shape[1]+1)
        self.xcart=np.reshape(cart_out[:,0],load_shape,order="C")
        self.ycart=np.reshape(cart_out[:,1],load_shape,order="C")
        self.gridx = RectBivariateSpline(np.arange(self.padded_im.shape[0]+1),np.arange(self.padded_im.shape[1]+1),self.xcart)
        self.gridy = RectBivariateSpline(np.arange(self.padded_im.shape[0]+1),np.arange(self.padded_im.shape[1]+1),self.ycart)

    def plot_density_map(self,include_transform=True):
        plt.imshow(self.padded_im,origin="lower")
        if include_transform:
            try:
                _ = self.xcart
            except:
                raise ValueError("Load data first")
            plt.plot(self.xcart,self.ycart,',',color="orange",alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.axis("equal")
            

    def calculate_grid_transforms(self):
        minx,miny,maxx,maxy=self.polygon_df.total_bounds
        #interp coordinates based on array indexing, opposite of image
        to_grid =[0,0,0,0,0,0]
        to_grid[0] = self.imsize[1] / (maxx - minx)
        to_grid[3] = self.imsize[0] / (maxy - miny)
        to_grid[4] = self.pad_width[1] - to_grid[0]*minx
        to_grid[5] = self.pad_width[0] - to_grid[3]*miny
        
        
        from_grid = [0,0,0,0,0,0]
        from_grid[0] = 1/to_grid[0]
        from_grid[3] = 1/to_grid[3]
        from_grid[4] = minx - from_grid[0]*self.pad_width[1]
        from_grid[5] = miny - from_grid[3]*self.pad_width[0]
        
        self.to_grid = to_grid
        self.from_grid = from_grid
        
    def get_cartogram_transformed_data(self):    
        def transform_point(p):
            px,py = p.coords[0]
            p = (self.gridx(py,px),self.gridy(py,px))#interp coordinates based on array indexing, opposite of image
            return p

        def transform_polygon(pg):
            pg = affine_transform(pg,self.to_grid)
            pg = Polygon([transform_point(p) for p in MultiPoint(pg.exterior.coords)])
            pg = affine_transform(pg,self.from_grid)
            return pg        
        def transform_polygon_or_multipolygon(x):
            if x.type == "MultiPolygon":
                return MultiPolygon([transform_polygon(pg) for pg in x])
            elif x.type == "Polygon":
                return transform_polygon(x)
            else:
                raise NotImplementedError(f"Geometry type : {x.type} is not implemented")
        #self.polygon_df["old geometry"] = self.polygon_df.geometry
        self.polygon_df = self.polygon_df.set_geometry(self.polygon_df.geometry.apply(transform_polygon_or_multipolygon))

    def _test_affine_transforms(self):
        to_and_from = lambda x: affine_transform( affine_transform(x, self.to_grid),self.from_grid)
        transformed_geometry = self.polygon_df.geometry.apply(to_and_from)
        return [sum(self.polygon_df.geom_almost_equals(transformed_geometry,decimal=d))for d in range(14)]
        

if __name__ == "__main__":
    mobility_data_fname = "/media/micha/DATA1/re_20200306.csv_10min.csv_bg_condensed.csv" if len(argv) < 2 else argv[1]
    polygon_data_fname = "/home/micha/devsnaps/cart-1.2.2/nyc_bg.geojson"
    cartogram_polygon_fname = "nyc_tag.geojson"
    mc = MobilityCartogram()
    log_step("Loading data")
    mc.data_df = pd.read_csv(mobility_data_fname,dtype={"bg":str,"timestamp":str,mc.data_key:float})
    mc.load_polygon_df(polygon_data_fname)
    mc.join_polygon_and_data()
    log_step("Preparing data for cart")
    #mc.create_image_from_geodf()
    #mc.image_to_density_map()
    mc.create_geocube_density_map()
    log_step("Running cart")
    mc.run_cart_on_density_map()
    log_step("Loading cart results")
    mc.load_interpolated_output()
    mc.calculate_grid_transforms()
    log_step("Transforming polygons")
    mc.get_cartogram_transformed_data()
    log_step("Saving new file")
    dftag =  mc.polygon_df.drop(columns=["old geometry"]).to_crs("EPSG:4326")
    call(["rm", cartogram_polygon_fname])
    dftag.to_file(cartogram_polygon_fname,driver="GeoJSON")
        
