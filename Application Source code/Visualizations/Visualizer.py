import matplotlib.pyplot as plt

import sys
sys.path.append('./Objects')
import Variables as vr

def plotHistogram():
    df = vr.dataFrameProcessed
    plt.figure(figsize=(7, 4.5))
    totalCount = len(df['Sentiment'])
    ax = df['Sentiment'].value_counts().plot(kind='bar')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel('Type of sentiments', labelpad=15, size = 14)
    plt.ylabel('Count of sentiments', labelpad=15, size = 14)
    plt.title('Distribution of sentiments (Total: ' + str(totalCount) + ')', pad=20, size = 18)
    plt.show()

def plotPieChart():
    df = vr.dataFrameProcessed
    totalCount = len(df['Sentiment'])
    # Create the pie chart
    plt.figure(figsize=(7, 5))
    counts = df['Sentiment'].value_counts()
    labels = counts.index
    sizes = counts.values

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', radius=0.5)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is circular.
    plt.title('Pie Chart of Sentiments (Total: ' + str(totalCount) + ')', pad=20, size = 18)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.show()