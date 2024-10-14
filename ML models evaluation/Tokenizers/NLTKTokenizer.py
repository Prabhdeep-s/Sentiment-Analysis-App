import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stopword_set = set(stopwords.words('english'))  
wnl = WordNetLemmatizer()
    
def preprocess_text_custom(text, returnTokens = True):
    text= re.sub(r"http\S+", "",text)
    text= re.sub(r"<.*?>", "",text)
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  
                        u"\U0001F300-\U0001F5FF"  
                        u"\U0001F680-\U0001F6FF"  
                        u"\U0001F1E0-\U0001F1FF"  
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    text= emoji_pattern.sub(r'',text)

    text= text.lower()
    text= re.sub(r"[^a-zA-Z]+", " ", text)
    
    text = word_tokenize(text)

    text = [wnl.lemmatize(token) for token in text if token not in stopword_set]

    if returnTokens:
        return text
    else:
        return " ".join(text)