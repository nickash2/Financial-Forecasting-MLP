from src.preprocess import preprocess, plot_preprocessed
import pandas as pd



















if __name__ == "__main__":
    data = preprocess()
    print(data.head())
    plot_preprocessed(data)