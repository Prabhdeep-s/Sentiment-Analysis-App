import re
import spacy

# Existing names: ['tok2vec', 'tagger', 'parser', 'senter', 'attribute_ruler', 'lemmatizer', 'ner']
nlp = spacy.load("en_core_web_sm", disable = ['tok2vec', 'parser', 'ner', 'senter'])

stop_words = nlp.Defaults.stop_words

def preprocess_text_spacy(text, returnTokens = True):
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

    doc = nlp(text)
    text = [ word.lemma_.lower().strip() for word in doc if word not in stop_words]
    text = [ word for word in text if word not in stop_words]

    if returnTokens:
        return text
    else:
        return " ".join(text)