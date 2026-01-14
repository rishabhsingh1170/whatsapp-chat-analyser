#import necessary libraries
import pandas as pd
import numpy as np
import re

def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'  #pattern to identify date and time in the chat
   
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    df = pd.DataFrame({'message_date':dates, 'user_messages':messages})

    df['message_date'] = pd.to_datetime(df['message_date'], format= '%d/%m/%y, %H:%M - ')

    df.rename(columns={'message_date':'date'},inplace = True)

    #seperating user and message
    users = []
    messages = []

    for message in df['user_messages']:
        entry = re.split(r'([^:]+):\s', message, maxsplit=1)

        if len(entry) == 3:
            users.append(entry[1].strip())
            messages.append(entry[2].strip())
        else:
            users.append('group_notification')
            messages.append(message.strip())

    df['user'] = users
    df['message'] = messages

    df.drop(columns='user_messages', inplace = True)

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['month_num'] = df['date'].dt.month
    df['day_name'] = df['date'].dt.day_name()

    period = []

    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df