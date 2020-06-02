from glob import glob
from matplotlib import pyplot as plt
from os.path import join,basename
import numpy as np
import datetime
import dateutil


def get_data_to_plot(dt,base_dir="/tmp/"):
    timestring =dt.isoformat()
    globstring = join(base_dir,f"mobility_cartogram_{timestring}*")
    density_fname, cartogram_ct_fname, _, cartogram_bg_fname, out_fname, density_transformed_fname = sorted(glob(globstring))
    im1 = np.loadtxt(density_fname)
    im2 = np.loadtxt(density_transformed_fname)
    cart_out = np.loadtxt(out_fname)
    load_shape = (im1.shape[0]+1,im1.shape[1]+1)
    xcart=np.reshape(cart_out[:,0],load_shape,order="C")
    ycart=np.reshape(cart_out[:,1],load_shape,order="C")
    global_avg = float("-".join(density_fname.split("_")[-1].split("-")[1:3]))
    weight=float(density_fname.strip(".dat").split("-")[-1])
    return im1,im2,xcart,ycart,global_avg,weight


def plot_data(dt,base_dir):
    im1,im2,xcart,ycart,global_avg,weight=get_data_to_plot(dt,base_dir)
    n=plt.Normalize(vmin=0,vmax=np.percentile(im1.flatten(),99))
    fig = plt.figure(frameon=False)
    dpi=300
    height,width=im1.shape
    fig.set_size_inches(width/dpi,height/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax,facecolor="black")
    ax.imshow(np.ma.masked_where(np.isclose(im1,im1[0][0],atol=1e-12),im1),origin="lower",aspect='equal',norm=n)
    ax.text(0.1, 0.9, dt.strftime("%a, %b %d, %I%p"), bbox=dict(facecolor='black', alpha=0.9),transform=ax.transAxes,color="darkgray")
    ofname=join(output_dir, subdir1, f"mobility_cartogram_{dt.isoformat}_original_data.png")
    fig.savefig(ofname, dpi=dpi)
    plt.close()
    
    fig = plt.figure(frameon=False)
    dpi=300
    height,width=im1.shape
    fig.set_size_inches(width/dpi,height/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im2*0,origin="lower",aspect="equal",cmap="inferno")
    ax.imshow(np.ma.masked_where(np.isclose(im1,im1[0][0],atol=1e-12),im1),origin="lower",aspect='equal',norm=n)
    ax.plot(xcart,ycart,',',color="orange",alpha=0.2)
    ax.text(0.1, 0.9, dt.strftime("%a, %b %d, %I%p"), bbox=dict(facecolor='black', alpha=0.9),transform=ax.transAxes,color="darkgray")
    ofname=join(output_dir, subdir1, f"mobility_cartogram_{dt.isoformat}_deform_on_original_data.png")
    fig.savefig(ofname, dpi=dpi)
    plt.close()
    
    fig = plt.figure(frameon=False)
    dpi=300
    height,width=im1.shape
    fig.set_size_inches(width/dpi,height/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax,facecolor="black")
    ax.imshow(im2*0,origin="lower",aspect="equal",cmap="inferno")
    ax.imshow(np.ma.masked_where(np.isclose(im2,im2[0][0],atol=1e-12),im2),origin="lower",aspect='equal',norm=n)
    ax.plot(xcart,ycart,',',color="orange",alpha=0.2)
    ax.text(0.1, 0.9, dt.strftime("%a, %b %d, %I%p"), bbox=dict(facecolor='black', alpha=0.9),transform=ax.transAxes,color="darkgray")
    ofname=join(output_dir, subdir1, f"mobility_cartogram_{dt.isoformat}_deform_on_deform_data.png")
    fig.savefig(ofname, dpi=dpi)
    plt.close()
    
    fig = plt.figure(frameon=False)
    dpi=300
    height,width=im1.shape
    fig.set_size_inches(width/dpi,height/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax,facecolor="black")
    ax.imshow(im2*0,origin="lower",aspect="equal",cmap="inferno")
    ax.plot(xcart,ycart,',',color="orange",alpha=0.2)
    ax.text(0.1, 0.9, dt.strftime("%a, %b %d, %I%p"), bbox=dict(facecolor='black', alpha=0.9),transform=ax.transAxes,color="darkgray")
    ofname=join(output_dir, subdir1, f"mobility_cartogram_{dt.isoformat}_deform_only.png")
    fig.savefig(ofname, dpi=dpi)
    plt.close()
    

if __name__ == "__main__":
    
    base_dir="/tmp/a"
    first_date = sorted(glob(join(base_dir,"mobility_cartogram*.dat")))[0]
    last_date = sorted(glob(join(base_dir,"mobility_cartogram*.dat")))[-1]
    first_date = dateutil.parser.parse(basename(first_date).split("_")[2])
    last_date = dateutil.parser.parse(basename(last_date).split("_")[2])
    
    this_date = first_date
    step = datetime.timedelta(hours=1)
    while this_date <= last_date:
        plot_data(this_date,base_dir)
        this_date += step
    
