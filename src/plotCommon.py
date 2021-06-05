import matplotlib.pyplot as plt

#linestyle
#https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html

def plot(x,y):
    plt.plot(x,y)
    plt.show()

# def plot(x,y=None):
#     if y:
#         plt.plot(x,y)
#     else:
#         plt.plot(x)
#     plt.show()
    
def plotSub(x,y,ax=None, aspect=False, label='', linestyle='solid', marker='',
            markersize=12, color=None):
    ax.plot(x,y,label=label,marker=marker,linestyle=linestyle,color=color)
    #ax.title.set_text(name)
    if aspect:
        ax.set_aspect(1)
    #ax.legend()

def scatterSub(x,y,ax=None,label='',marker='.'):
    ax.scatter(x,y,linewidths=.3, label=label, marker=marker)
    ax.legend()
    

