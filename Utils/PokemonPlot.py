import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker

typeNameToColorDict = {
    "normal":"#a8a77a",
    "fighting":"#bf2f2b",
    "poison":"#9e449e",
    "ground":"#e9be6c",
    "flying":"#a793ee",
    "bug":"#a9b62f",
    "rock":"#b89e3f",
    "ghost":"#6f5a97",
    "steel":"#b8b9cf",
    "fire":"#ef7e37",
    "water":"#6794ee",
    "electric":"#f8ce3f",
    "grass":"#7bc656",
    "ice":"#99d8d8",
    "psychic":"#f65988",
    "dragon":"#6d47f5",
    "dark":"#705849",
    "fairy":"#feb9f9",
}

typeNumberToNameDict = {
    1: 'normal',
    2: 'fighting',
    3: 'flying',
    4: 'poison',
    5: 'ground',
    6: 'rock',
    7: 'bug',
    8: 'ghost',
    9: 'steel',
    10: 'fire',
    11: 'water',
    12: 'grass',
    13: 'electric',
    14: 'psychic',
    15: 'ice',
    16: 'dragon',
    17: 'dark',
    18: 'fairy'
}

def numToColor(n):
    return typeNameToColorDict[typeNumberToNameDict[n]]

def plotSprite(f, ax, x, y1, y2=None):
#     ax.set_title(typeNumberToNameDict[y1].capitalize(), color=numToColor(y1), fontweight='bold')
    r, c, _ = x.shape
    buf = 5
    if y2 is None or y2 <= 0:
        ax.text(c/2, 0-buf, typeNumberToNameDict[y1].capitalize(), fontweight='bold', 
                ha="center", va="bottom", size="large",color=numToColor(y1))
    else:
        t0 = ax.text(0, 0, typeNumberToNameDict[y1].capitalize(), fontweight='bold', 
                ha="left", va="bottom", size="large",color=numToColor(y1))
        t1 = ax.text(0, 0, "  /  ", fontweight='bold', 
                ha="left", va="bottom", size="large")
        t2 = ax.text(0, 0, typeNumberToNameDict[y2].capitalize(), fontweight='bold', 
                ha="left", va="bottom", size="large",color=numToColor(y2))
        r = f.canvas.get_renderer()
        w0 = t0.get_window_extent(renderer=r).width
        w1 = t1.get_window_extent(renderer=r).width
        w2 = t2.get_window_extent(renderer=r).width
        total = w0 + w1 + w2
        t0.set_position(((c - total)/2, 0-buf))
        t1.set_position(((c - total)/2 + w0, 0-buf))
        t2.set_position(((c - total)/2 + w0 + w1, 0-buf))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(x)

def plotPredictions(x, y, scores, y2=None, k=5):
    N = x.shape[0]
    for i in range(N):
        f = plt.figure(figsize=(9,2.5))
        gs = gridspec.GridSpec(1,2)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
    
        plotSprite(f, ax0, x[i], y[i], y2=y2[i])
                              
        top_types_inds = np.argsort(scores[i])[::-1][:k]
        top_types = top_types_inds + 1
        inds = np.arange(k)
        labels = [typeNumberToNameDict[n].capitalize() for n in top_types]
        labels.reverse()
        ax1.yaxis.set_ticks(inds)
        ax1.set_yticklabels(labels)
        ax1.set_xlim((0,1))
        ax1.barh(inds[::-1], scores[i][top_types_inds], color=[numToColor(n) for n in top_types])
                              
def plotConfusionMatrix(mat, title, stats={}):
    f = plt.figure(figsize=(6,6))
    ax = f.add_subplot(111)
    labels = [typeNumberToNameDict[n].capitalize() for n in range(1,19)]
    labels.insert(0,"[blank]")
    
    statsString = ""
    for key in stats.keys():
        statsString += key + "=" + str(stats[key]) + ", "
    if len(statsString) > 1:
        statsString = statsString[:-2]

    cax = ax.matshow(mat)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(title + " Confusion Matrix (n=" + str(int(np.sum(mat))) + ")\n\nActual", fontsize=14)
    ax.set_ylabel("Predicted", fontsize=14)
    f.colorbar(cax)
    
def plotAccuracyAndLoss(h):
    # summarize history for accuracy
    plt.plot(h['acc'])
    plt.plot(h['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    