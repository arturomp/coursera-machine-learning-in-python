import matplotlib.pyplot as plt

def drawLine(p1, p2, **kwargs):
#DRAWLINE Draws a line from point p1 to point p2
#   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
#   current figure

    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)
