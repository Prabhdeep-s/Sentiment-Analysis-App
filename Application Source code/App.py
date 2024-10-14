import customtkinter
from tkinter import *
from tkinter import messagebox

import os
import sys
sys.path.append('./Classifier')
sys.path.append('./Objects')
sys.path.append('./ReviewAnalyzer')
sys.path.append('./TopicModelling')
sys.path.append('./Vectorizer')
sys.path.append('./Visualizations')
import Analyzer as anlyzer
import Visualizer as vs
import KeyThemesGenerator as ktg
import WordDensityGenerator as wdg
import Constants as cn
import Variables as vr
import pandas as pd


#imports for other project files
from sklearn.model_selection import train_test_split
from sklearn.linear_model._logistic import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from fasttext import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
from fasttext import load_model

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from collections import Counter

from wordcloud import WordCloud
from PIL import Image

import matplotlib.pyplot as plt
#end of imports

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("520x410")
app.title(cn.APP_TITLE)
app.resizable(False, False)
sentiment_label = StringVar()
file_uploaded_label = StringVar()

def open_Help_Window():
    helpWindow = customtkinter.CTkToplevel(app)
    helpWindow.geometry("455x340")
    helpWindow.title(cn.HELP_WINDOW_TITLE)
    helpWindow.resizable(False, False)
    helpFrame = customtkinter.CTkFrame(helpWindow, width=445, height=320, border_width=2)
    helpFrame.pack(side='left', fill='both', pady=10, padx=10, expand=True)
    label_HP = customtkinter.CTkLabel(helpFrame, text=cn.HELP_TEXT, justify=customtkinter.LEFT, font=customtkinter.CTkFont(size=14))
    label_HP.pack(pady=5, padx=10)

frame_1 = customtkinter.CTkFrame(master=app)
frame_1.pack(pady=10, padx=10, fill="both", expand=True)

frame_Help = customtkinter.CTkFrame(frame_1, bg_color="transparent", fg_color="transparent")
frame_Help.pack(pady=1, padx=1, fill="both")

help_button = customtkinter.CTkButton(frame_Help, text="How to?", width=50, height=20, font=customtkinter.CTkFont(size=12), command=open_Help_Window)
help_button.pack(side='right', pady=1, padx=2)

tabview_1 = customtkinter.CTkTabview(master=frame_1, width=480, height=380)
tabview_1.pack(pady=5, padx=10)
tabview_1.add(cn.TAB1_NAME)
tabview_1.add(cn.TAB2_NAME)

def open_file():
    file_types = [('Excel Files', '*.xlsx *.xls'), ('CSV Files', '*.csv')]
    vr.excelFilePath = customtkinter.filedialog.askopenfilename(title=cn.OPENFILETITLE, filetypes=file_types)
    if vr.excelFilePath:
        fileName = os.path.basename(vr.excelFilePath)
        if len(fileName)>25:
            fileName = "..." + str(fileName[-25:])
        file_uploaded_label.set(fileName)

def save_file(resultsDF):
    file_type = [('Excel Files', '*.xlsx *.xls'), ('CSV Files', '*.csv')]
    file_path = customtkinter.filedialog.asksaveasfilename(initialfile=cn.SAVEFILENAME, title=cn.SAVEFILETITLE, filetypes=file_type)

    if file_path!=None:
        # Get the file extension
        ext = os.path.splitext(str(file_path))[1]
        if ext == '.csv':
            resultsDF.to_csv(file_path, index=False)
        elif ext == '.xlsx' or ext == '.xls':
            resultsDF.to_excel(file_path, index=False)

def get_Sentiment_For_Text():
    sentiment_label.set("Sentiment = " + str(anlyzer.analyze_text(text_2.get('1.0', END))))

def get_Sentiments_For_File():
    if (vr.excelFilePath == None):
        messagebox.showwarning(cn.UPLOAD_WARNING_TILE,cn.UPLOAD_WARNING_MESSAGE)
    else:
        processedresults = anlyzer.analyze_file(vr.excelFilePath)
        if (processedresults.empty):
            messagebox.showwarning(cn.NO_REVIEWS_WARNING_TITLE, cn.NO_REVIEWS_WARNING_MESSAGE)
            return
        vr.IsReviewsProcessingDone = True
        save_file(processedresults)

def open_Words_Window():
    if (vr.IsReviewsProcessingDone == False):
        messagebox.showwarning(cn.ANALYSIS_WARNING_TILE, cn.ANALYSIS_WARNING_MESSAGE)
    else:
        wordWindow = customtkinter.CTkToplevel(app)
        wordWindow.geometry("640x360")
        wordWindow.title(cn.WORDS_WINDOW_TITLE)
        wordWindow.resizable(False, False)
        buttonsFrame = customtkinter.CTkFrame(wordWindow, width=100, height=380, border_width=2)
        buttonsFrame.pack(side='left', fill='both', pady=10, padx=10)
        global visualFrame
        visualFrame = customtkinter.CTkFrame(wordWindow, width=560, height=380, border_width=2)
        visualFrame.pack(side='left', fill='both', pady=10, padx=(5,10))
        button_HP = customtkinter.CTkButton(buttonsFrame, text=cn.SENTIMENT1, height=30, width=80, command= lambda: show_Word_Density(cn.SENTIMENT1))
        button_HP.pack(pady=(35,15), padx=5)
        button_P = customtkinter.CTkButton(buttonsFrame, text="      "+cn.SENTIMENT2+"      ", height=30, width=80, command= lambda: show_Word_Density(cn.SENTIMENT2))
        button_P.pack(pady=15, padx=5)
        button_Neu = customtkinter.CTkButton(buttonsFrame, text="       "+cn.SENTIMENT3+"       ", height=30, width=80, command= lambda: show_Word_Density(cn.SENTIMENT3))
        button_Neu.pack(pady=15, padx=5)
        button_N = customtkinter.CTkButton(buttonsFrame, text="      "+cn.SENTIMENT4+"      ", height=30, width=80, command= lambda: show_Word_Density(cn.SENTIMENT4))
        button_N.pack(pady=15, padx=5)
        button_HN = customtkinter.CTkButton(buttonsFrame, text=cn.SENTIMENT5, height=30, width=80, command= lambda: show_Word_Density(cn.SENTIMENT5))
        button_HN.pack(pady=15, padx=5)
        button_HP.focus_force()  

def show_Word_Density(sentimentType):
    word_Density_Image = wdg.getWordDensity(sentimentType)
    if word_Density_Image == None:
        messagebox.showinfo(cn.NO_WORDS_WARNING_TITLE, cn.NO_WORDS_WARNING_MESSAGE)
    else:
        ctkImage = customtkinter.CTkImage(light_image=word_Density_Image, size=(460,340))
        for widgets in visualFrame.winfo_children():
            widgets.destroy()
        word_cloud_label = customtkinter.CTkLabel(visualFrame, image=ctkImage, text="")
        word_cloud_label.pack()

def show_Histogram():
    if (vr.IsReviewsProcessingDone == False):
        messagebox.showwarning(cn.ANALYSIS_WARNING_TILE, cn.ANALYSIS_WARNING_MESSAGE)
    else:
        vs.plotHistogram()

def show_PieChart():
    if (vr.IsReviewsProcessingDone == False):
        messagebox.showwarning(cn.ANALYSIS_WARNING_TILE, cn.ANALYSIS_WARNING_MESSAGE)
    else:
        vs.plotPieChart()

def open_KeyThemes_Window():
    if (vr.IsReviewsProcessingDone == False):
        messagebox.showwarning(cn.ANALYSIS_WARNING_TILE, cn.ANALYSIS_WARNING_MESSAGE)
    else:
        keyTheme_HN, keyTheme_N, keyTheme_Neu, keyTheme_P, keyTheme_HP = ktg.getKeyThemes()

        themesWindow = customtkinter.CTkToplevel(app)
        themesWindow.geometry("640x420")
        themesWindow.title(cn.WORDS_WINDOW_TITLE)
        themesWindow.resizable(False, False)
        themesFrame = customtkinter.CTkFrame(themesWindow, width=620, height=400)
        themesFrame.pack(fill='both', pady=5, padx=5)

        vFrameHP = customtkinter.CTkFrame(themesFrame, border_width=2)
        vFrameHP.pack(fill='both', pady=5, padx=5)
        label_HP = customtkinter.CTkLabel(vFrameHP, text=cn.THEME_LABEL_HP, justify=customtkinter.LEFT, font=customtkinter.CTkFont(underline=True))
        label_HP.pack(pady=5, padx=10)
        label_HP_Theme = customtkinter.CTkLabel(vFrameHP, text=keyTheme_HP, justify=customtkinter.LEFT, font=customtkinter.CTkFont(size=14))
        label_HP_Theme.pack(pady=5, padx=5)
        
        vFrameP = customtkinter.CTkFrame(themesFrame, border_width=2)
        vFrameP.pack(fill='both', pady=5, padx=5)
        label_P = customtkinter.CTkLabel(vFrameP, text=cn.THEME_LABEL_P, justify=customtkinter.LEFT, font=customtkinter.CTkFont(underline=True))
        label_P.pack(pady=5, padx=10)
        label_P_Theme = customtkinter.CTkLabel(vFrameP, text=keyTheme_P, justify=customtkinter.LEFT, font=customtkinter.CTkFont(size=14))
        label_P_Theme.pack(pady=(0,5), padx=5)

        vFrameNeu = customtkinter.CTkFrame(themesFrame, border_width=2)
        vFrameNeu.pack(fill='both', pady=5, padx=5)
        label_Neu = customtkinter.CTkLabel(vFrameNeu, text=cn.THEME_LABEL_NEU, justify=customtkinter.LEFT, font=customtkinter.CTkFont(underline=True))
        label_Neu.pack(pady=5, padx=10)
        label_Neu_Theme = customtkinter.CTkLabel(vFrameNeu, text=keyTheme_Neu, justify=customtkinter.LEFT, font=customtkinter.CTkFont(size=14))
        label_Neu_Theme.pack(pady=(0,5), padx=5)

        vFrameN = customtkinter.CTkFrame(themesFrame, border_width=2)
        vFrameN.pack(fill='both', pady=5, padx=5)
        label_N = customtkinter.CTkLabel(vFrameN, text=cn.THEME_LABEL_N, justify=customtkinter.LEFT, font=customtkinter.CTkFont(underline=True))
        label_N.pack(pady=5, padx=10)
        label_N_Theme = customtkinter.CTkLabel(vFrameN, text=keyTheme_N, justify=customtkinter.LEFT, font=customtkinter.CTkFont(size=14))
        label_N_Theme.pack(pady=(0,5), padx=5)

        vFrameHN = customtkinter.CTkFrame(themesFrame, border_width=2)
        vFrameHN.pack(fill='both', pady=5, padx=5)
        label_HN = customtkinter.CTkLabel(vFrameHN, text=cn.THEME_LABEL_HN, justify=customtkinter.LEFT, font=customtkinter.CTkFont(underline=True))
        label_HN.pack(pady=5, padx=10)
        label_HN_Theme = customtkinter.CTkLabel(vFrameHN, text=keyTheme_HN, justify=customtkinter.LEFT, font=customtkinter.CTkFont(size=14))
        label_HN_Theme.pack(pady=(0,5), padx=5)

        
# TAB 1 controls begin
tab1 = tabview_1.tab(cn.TAB1_NAME)

label_1 = customtkinter.CTkLabel(tab1, text=cn.LABEL1_TEXT, justify=customtkinter.LEFT)
label_1.pack(pady=(18,5), padx=10)

frame_File = customtkinter.CTkFrame(master=tab1)
frame_File.pack(pady=5, padx=5, fill="both")

label_File = customtkinter.CTkLabel(frame_File, text="File name:", justify=customtkinter.LEFT)
label_File.pack(side='left', pady=15, padx=5)

file_uploaded_result = customtkinter.CTkLabel(frame_File, justify=customtkinter.LEFT, textvariable = file_uploaded_label, width=210, height=28,
                                              bg_color='gray44')
file_uploaded_result.pack(side='left', pady=15, padx=5)

open_button = customtkinter.CTkButton(frame_File, text="Upload a File", command=open_file)
open_button.pack(side='left', pady=15, padx=8)

button_1 = customtkinter.CTkButton(master=tab1, text=cn.BUTTON1_TEXT, width=200, height=35, command=get_Sentiments_For_File)
button_1.pack(pady=(15,30), padx=10)

frame_Vis = customtkinter.CTkFrame(master=tab1)
frame_Vis.pack(pady=5, padx=5, fill="both", expand=True)

label_Vis = customtkinter.CTkLabel(frame_Vis, text=cn.LABEL_VIS, justify=customtkinter.LEFT)
label_Vis.pack(pady=(10,5), padx=10)

button_Hist = customtkinter.CTkButton(master=frame_Vis, text="Histogram", command=show_Histogram, width=90)
button_Hist.pack(side='left', pady=(5,10), padx=(12,8))

button_Pie = customtkinter.CTkButton(master=frame_Vis, text="Pie-chart", command=show_PieChart, width=90)
button_Pie.pack(side='left', pady=(5,10), padx=8)

button_WordsDensity = customtkinter.CTkButton(master=frame_Vis, text="Word Density", command=open_Words_Window, width=90)
button_WordsDensity.pack(side='left', pady=(5,10), padx=8)

button_KeyThemes = customtkinter.CTkButton(master=frame_Vis, text="Key themes", command=open_KeyThemes_Window, width=90)
button_KeyThemes.pack(side='left', pady=(5,10), padx=8)

# TAB 1 controls end

# TAB 2 controls begin
tab2 = tabview_1.tab(cn.TAB2_NAME)

label_2 = customtkinter.CTkLabel(master=tab2, text=cn.LABEL2_TEXT, justify=customtkinter.LEFT)
label_2.pack(pady=(15,5), padx=10)

text_2 = customtkinter.CTkTextbox(master=tab2, width=360, height=140)
text_2.pack(pady=(5,10), padx=10)
text_2.insert("0.0", "")

button_2 = customtkinter.CTkButton(master=tab2, text=cn.BUTTON2_TEXT, command=get_Sentiment_For_Text)
button_2.pack(pady=(15,10), padx=10)

label_result = customtkinter.CTkLabel(master=tab2, text="NA", justify=customtkinter.LEFT, textvariable = sentiment_label)
label_result.pack(pady=10, padx=10)

# TAB 2 controls end

app.mainloop()