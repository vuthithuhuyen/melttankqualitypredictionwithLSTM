from matplotlib import pyplot as plt
import pandas as pd


def PandasColToPlot(df: pd.DataFrame, col, color):
    plt.figure()
    plt.plot(df[df.columns[col]], label=df.columns[col], color=color)
    plt.xlabel("Samples")
    plt.ylabel("Temprature")
    plt.grid()
    plt.legend()
    return plt
# ---------------------------------------------

def PandasColToHist(df: pd.DataFrame, col, color):
    plt.figure()
    plt.hist(df[df.columns[col]], facecolor=color)
    # plt.xlabel("Samples")
    # plt.ylabel("Temprature")
    plt.xticks(rotation=45)
    plt.title(df.columns[col])
    return plt
# ---------------------------------------------