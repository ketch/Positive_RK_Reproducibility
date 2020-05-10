#!/usr/bin/python3
#Code from https://ranocha.de/blog/colors/#python-code under MIT-Licence (https://ranocha.de/assets/code_for_posts/LICENSE.txt)
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.path import Path


def setup_plt():
    # line cyclers adapted to colourblind people
    line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
    marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

                 # matplotlib's standard cycler
    standard_cycler = cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])

    plt.rc("axes", prop_cycle=line_cycler)

    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}")
    plt.rc("font", family="serif", size=18.)
    plt.rc("savefig", dpi=200)
    plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
    plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)


def circle(r_in,r_out):
    '''
    Function to generate the path of a circle with a hole. Used to genreate a o-shaped marker. Based on Path.circle
    '''
    center = (0,0)
    MAGIC = 0.2652031
    SQRTHALF = np.sqrt(0.5)
    MAGIC45 = SQRTHALF * MAGIC

    vertices_out = np.array([[0.0, -1.0], #start |end

                             [MAGIC, -1.0],
                             [SQRTHALF-MAGIC45, -SQRTHALF-MAGIC45],
                             [SQRTHALF, -SQRTHALF],

                             [SQRTHALF+MAGIC45, -SQRTHALF+MAGIC45],
                             [1.0, -MAGIC],
                             [1.0, 0.0],

                             [1.0, MAGIC],
                             [SQRTHALF+MAGIC45, SQRTHALF-MAGIC45],
                             [SQRTHALF, SQRTHALF],

                             [SQRTHALF-MAGIC45, SQRTHALF+MAGIC45],
                             [MAGIC, 1.0],
                             [0.0, 1.0],

                             [-MAGIC, 1.0],
                             [-SQRTHALF+MAGIC45, SQRTHALF+MAGIC45],
                             [-SQRTHALF, SQRTHALF],

                             [-SQRTHALF-MAGIC45, SQRTHALF-MAGIC45],
                             [-1.0, MAGIC],
                             [-1.0, 0.0],

                             [-1.0, -MAGIC],
                             [-SQRTHALF-MAGIC45, -SQRTHALF+MAGIC45],
                             [-SQRTHALF, -SQRTHALF],

                             [-SQRTHALF+MAGIC45, -SQRTHALF-MAGIC45],
                             [-MAGIC, -1.0],
                             [0.0, -1.0]#end|start
                    ],
                            dtype=float)

    vertices_con = np.array([[0.,-1.],[0.,-1.]],dtype=float)
    vertices_in = np.flip(vertices_out,0)

    codes_out = [Path.CURVE4] * 25
    codes_in = [Path.CURVE4] * 25
    #codes = [Path.LINETO] *26
    codes_out[0] = Path.MOVETO
    codes_in[0] = Path.LINETO

    codes_con = [Path.LINETO,Path.CLOSEPOLY]


    return Path(np.concatenate((vertices_out * r_out + center,
                                vertices_in * r_in + center,
                                vertices_con * r_out + center),axis=0),
                np.concatenate((codes_out,codes_in,codes_con)), readonly=True)


def marker(num):
    if num == -1: #default
        return '.'
    if num == 0:
        return circle(0.5,1)
    elif num==1:
        return 10
    elif num ==2:
        return (num, 2, 0)
    elif num == 3:
        return (num, 2, 180)
    elif num == 4:
        return (num, 2, 45)
    else:
        return (num, 2, 0)

def plot_markers(x,y,default,num,color,label = True,zorder =2):
    print(x.shape)
    print(y.shape)
    print(default.shape)
    print(num.shape)
    """
    Plots markers depending on the input
    x:  x-coordiantes
    y:  y-coordinates
    default: boolen mask for default (as nparray)
    num: ordes for num (as nparray)
    color: color of markers
    """
    if label:
        plt.scatter(x[default],y[default],color=color,marker=marker(-1),label='$\\tilde{b}=b$',zorder=zorder)
    else:
        plt.scatter(x[default],y[default],color=color,marker=marker(-1),zorder=zorder)
    for o in range(0,5):
        #print(o,mask)
        mask = (num == o)
        if np.any(mask):
            if label:
                plt.scatter(x[mask],y[mask],color=color,marker=marker(o),label= 'Order= '+str(o),zorder=zorder)
            else:
                plt.scatter(x[mask],y[mask],color=color,marker=marker(o),zorder=zorder)
