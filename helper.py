from urlextract import URLExtract
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter 
import pandas as pd
import emoji
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

extractor = URLExtract()

def fetch_stats(selected_user, df):
    new_df = df
    if selected_user != 'overall':
        new_df = df[df['user'] == selected_user]

    #fetching total messages
    no_messages = new_df.shape[0]
    words = []
    # fetching total words
    for message in df['message']:
        if message != '<Media omitted>':
            words.extend(message.split())  

    # fetching no of media
    no_of_media = new_df[df['message'] == '<Media omitted>'].shape[0] 

    #fetching no of links     
    links =[]

    for message in new_df['message']:
        link = extractor.find_urls(message)
        if len(link):
            links.append(link)

    return no_messages, len(words), no_of_media, len(links)

def fetch_busy_user(df):
    df = df[df['user'] != 'group_notification']
    df = df[df['user'] != 'Meta AI']

    x = df['user'].value_counts().head()
    df = round(df['user'].value_counts()/df.shape[0] * 100, 2).reset_index().rename(columns={'user' : 'name', 'count':'percentage'})
    return x, df

def create_wordCloud(selected_user, df):
    new_df = df
    if selected_user != 'overall':
        new_df = df[df['user'] == selected_user]

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(df[df['message'] != '<Media omitted>']['message'].astype(str).str.cat(sep = ' '))
    return df_wc    

def fetch_most_common_words(selected_user, df):
    new_df = df
    if selected_user != 'overall':
        new_df = df[df['user'] == selected_user]

    temp = new_df[new_df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']

    clean_message = temp['message'].apply(remove_stopwords)
    all_words = [word for words in clean_message for word in words]
    return Counter(all_words).most_common(20)

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return [w for w in words if w not in stop_words]

def emoji_helper(selected_user, df):
    new_df = df
    if selected_user != 'overall':
        new_df = df[df['user'] == selected_user]

    emojis = []

    for message in new_df['message']:
        if isinstance(message, str):   
            emojis.extend([c for c in message if c in emoji.EMOJI_DATA]) 
    return pd.DataFrame(Counter(emojis).most_common(5))       

def monthly_timeLine(selected_user, df):
    new_df = df
    if selected_user != 'overall':
        new_df = df[df['user'] == selected_user]

    timeLine = new_df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()  

    time = []

    for i in range(timeLine.shape[0]):
        time.append(timeLine['month'][i] + '-' + str(timeLine['year'][i]))  

    timeLine['time'] = time

    return timeLine    

def most_busy_day(selected_user, df):
    new_df = df
    if selected_user != 'overall':
        new_df = df[df['user'] == selected_user]

    return new_df['day_name'].value_counts()    

def most_month_day(selected_user, df):
    new_df = df
    if selected_user != 'overall':
        new_df = df[df['user'] == selected_user]

    return new_df['month'].value_counts() 

def activity_heatMap(selected_user, df):
    user_heatMap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatMap    

def find_similarity_user_msg(selected_user, df):

    df = df[df['user'] != 'group_notification']
    df = df[df['message'] != '<Media omitted>']
    df = df[df['user'] != 'Meta AI']

    df = df[(~df['message'].str.contains('deleted', case= False, na = False))&(~df['message'].str.contains(r'https+', regex=True, na=False))]
    texts = df.groupby('user')['message'].apply(' '.join).to_dict()

    users = list(texts.keys())
    user_text = list(texts.values())

    vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=2,              # ignore very rare words
        ngram_range=(1, 2),    # capture phrases
        max_features=5000
    )

    tfid = vectorizer.fit_transform(user_text)

    similarity_matrix = cosine_similarity(tfid)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index= users,
        columns=users
    )

    return similarity_df