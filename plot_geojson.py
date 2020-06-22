import geopandas as gpd
import matplotlib.pyplot as plt
from sys import argv
from os.path import basename,join
import dateutil
from multiprocessing import Pool
from glob import glob

working_dir = argv[1]
globstring = argv[2]
plt.switch_backend("Agg")
fig_w,fig_h = (3,10)


def plot_geojson(fname):
    gdf = gpd.read_file(fname).to_crs( "EPSG:6539")
    try:
        origin=gdf.unary_union.centroid
    except:
        gdf= gdf.set_geometry(gdf.geometry.apply(lambda x: x.buffer(0)))
        origin=gdf.unary_union.centroid
    gdf = gdf.set_geometry(gdf.rotate(28.5, origin=origin))

    f, a = plt.subplots(figsize=(fig_w,fig_h),frameon=False)
    ts=basename(fname).split("_")[2]
    print(ts)
    dt = dateutil.parser.parse(ts)

    gdf.plot(ax=a,facecolor="none",edgecolor="black",linewidth=1)
    a.text(0.1, 0.99, dt.strftime("%a, %b %d, %I%p"), bbox=dict(facecolor='black', alpha=0.9),transform=a.transAxes,color="darkgray",fontdict={"size":10})            
            
    a.axis("off")
    ofname=".".join(basename(fname).split(".")[:-1])+".png"
    plt.savefig(ofname,dpi=300)
    plt.close()
    return 1

with Pool() as pool:
    res = pool.map_async(plot_geojson,sorted(glob(join(working_dir,globstring))))
    print(sum(res.get()))
