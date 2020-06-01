from glob import glob
from matplotlib import pyplot as plt
from os.path import join
import numpy as np
import datetime



def get_data_to_plot(y,m,d,h,base_dir="/tmp/"):
    timestring = datetime.datetime(y,m,d,h).isoformat()
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



if __name__ == "__main__":
    y,m,d,h=(2020,3,8,14)
    base_dir="/tmp/a"
    dt = datetime.datetime(y,m,d,h)
    
    
    im1,im2,xcart,ycart,global_avg,weight=get_data_to_plot(y,m,d,h,base_dir)
    n=plt.Normalize(vmin=0,vmax=2*global_avg)
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
    fig.savefig('figure.png', dpi=dpi)
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
    fig.savefig('figure1.png', dpi=dpi)
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
    fig.savefig('figure2.png', dpi=dpi)
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
    fig.savefig('figure3.png', dpi=dpi)
    plt.close()
        
