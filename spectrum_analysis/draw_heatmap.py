


#################################################
# draw heatmap for matrix

# Author : ghzhang
# Date   :
# HomePage :
# Email  : 
#################################################


from matplotlib import cm
#from pyheatmap.heatmap import HeatMap
import matplotlib.pyplot as plt




def draw_heatmap(data,xlabels,ylabels):
    cmap = cm.jet
    figure=plt.figure(facecolor='w')
    ax=figure.add_subplot(2,1,1,position=[0.1,0.15,0.8,0.8])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    vmax=data[0][0]
    vmin=data[0][0]
    for i in data:
        for j in i:
            if j>vmax:
                vmax=j
            if j<vmin:
                vmin=j
    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
    cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)

    plt.show()
