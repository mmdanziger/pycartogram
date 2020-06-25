from pycartogram import Cartogram
import argparse
import geopandas as gpd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformed-density-fname')
    parser.add_argument('--cartogram-metadata-fname')
    parser.add_argument('--input-geojson-fname')
    parser.add_argument('--rotate-angle',type=float,default=28.5)
    parser.add_argument('--output-geojson-fname')
    args = parser.parse_args()
    
    mc = Cartogram()
    mc.load_transform_data()
    mc.transformed_density_fname = args.transformed_density_fname
    mc.polygon_df = gpd.read_file(args.input_geojson_fname).to_crs(mc.projection)
    try:
        origin = mc.polygon_df.unary_union.centroid
    except:
        mc.polygon_df = mc.polygon_df.set_geometry(mc.polygon_df.geometry.apply(lambda x: x.buffer(0)))
        origin = mc.polygon_df.unary_union.centroid
        
    mc.polygon_df = mc.polygon_df.set_geometry(mc.polygon_df.rotate(angle=args.rotate_angle,origin=origin))
    mc.get_cartogram_transformed_data()
    dftag =  mc.polygon_df.set_geometry(mc.polygon_df.rotate(-28.5,origin=pivot_point)).to_crs("EPSG:4326")
    call(["rm", cartogram_polygon_fname])
    dftag.to_file(cartogram_polygon_fname,driver="GeoJSON")
    
if __nmae__ == "__main__":
    main()
                                               
