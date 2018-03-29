import pandas as pd
import matplotlib.pyplot as plt

def plot(df):
    plt.figure(dpi=300)
    df.plot(legend=False, alpha=0.3, figsize=(20, 10), color='grey', ax=plt.gca())
    linewidth = 5
    df['China'].plot(linewidth=linewidth, color='#f44336', legend=True, alpha=0.7)
    df['United Kingdom'].plot(
        linewidth=linewidth, color='#2196F3', legend=True, alpha=0.7)
    df['United States'].plot(
        linewidth=linewidth, color='#4CAF50', legend=True, alpha=0.7)
    df['Niger'].plot(linewidth=linewidth, color='#795548', legend=True, alpha=0.7)
    df['Yemen, Rep.'].plot(linewidth=linewidth,
                        color='#212121', legend=True, alpha=0.7)
    df['India'].plot(linewidth=linewidth, color='#FF9800', legend=True, alpha=0.7)
    plt.xlabel('Year')
    plt.title('Fertility Rates (birth per woman)')
    plt.savefig('Fertility Rates.png')

if __name__ == '__main__':
    df = pd.read_csv('Fertility-Rates-processed.csv')
    df = df.set_index('Unnamed: 0')
    plot(df)
    print('END OFFF SCRIPT.')
    