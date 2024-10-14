from wordcloud import WordCloud
from PIL import Image
import pandas as pd

import sys
sys.path.append('./Objects')
import Variables as vr

def getWordDensity(sentimentType):
    filtered_df = vr.dataFrameProcessed[vr.dataFrameProcessed['Sentiment'] == sentimentType]
    if filtered_df.empty:
        return None
    else:
        concatenated_Reviews = ''.join(filtered_df['TokenizedReviews'].tolist())
        wordcloud = WordCloud(max_font_size=50, width=460, height=340, max_words=20, background_color="white").generate(concatenated_Reviews)
        image = Image.fromarray(wordcloud.to_array())
        return image
