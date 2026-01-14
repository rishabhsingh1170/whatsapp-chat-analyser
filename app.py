import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("What's Chat Analyser")

uploaded_file = st.sidebar.file_uploader("choose a file")
if uploaded_file is not None :
    byte_data = uploaded_file.getvalue()
    data = byte_data.decode('utf-8')
    # st.text(data)
    df = preprocessor.preprocess(data)
    # st.dataframe(df)    

    #fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,'overall')

    selected_user = st.sidebar.selectbox("Show Anylises wrt : ", user_list)    

    if st.sidebar.button("Show Analysis"):

        #stats Area
        no_messages, words, no_of_media, no_of_links = helper.fetch_stats(selected_user, df)

        st.title("Top Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header('Total messages')
            st.title(no_messages)
        with col2:
            st.header('Total Words')
            st.title(words)
        with col3:
            st.header('Total Media')
            st.title(no_of_media)    
        with col4:
            st.header('Total links')
            st.title(no_of_links) 

        #timeLine
        st.title("Monthly Timeline")
        timeLine = helper.monthly_timeLine(selected_user, df)

        fig, ax = plt.subplots()
        ax.plot(timeLine['time'], timeLine['message'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig) 

        #Activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1 :
            st.header("Most Busy Day")
            busy_day = helper.most_busy_day(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)   

        with col2 :
            st.header("Most Busy Month")
            busy_month = helper.most_month_day(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color = 'orange')
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig) 

        #heatMap
        st.title("Weekly Activity Map")
        userHeatMap = helper.activity_heatMap(selected_user, df)

        fig, ax = plt.subplots()
        ax = sns.heatmap(userHeatMap)
        st.pyplot(fig)     


        #find busiest user in group
        if selected_user == 'overall':
            st.title("most busy users")
            x, new_df = helper.fetch_busy_user(df)
            fig, ax = plt.subplots()
            
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color = 'red')
                plt.xticks(rotation = 'vertical')
                st.pyplot(fig) 
            with col2:
                st.dataframe(new_df)   

        #wordCloud
        df_wc = helper.create_wordCloud(selected_user, df)
        st.title("WordCloud")
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig) 

        #most common words
        most_common_words = helper.fetch_most_common_words(selected_user, df)
        
        words = [w for w, c in most_common_words]
        counts = [c for w, c in most_common_words]
        
        fig, ax = plt.subplots()
        ax.bar(words, counts)
        plt.xticks(rotation = 'vertical')
        st.title("Most Common Words")
        st.pyplot(fig)
         


        #Emoji Analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Top 5 Emojis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(emoji_df)
        with col2:                         
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1], labels=emoji_df[0], autopct='%0.2f')
            st.pyplot(fig)

        # User Message Similarity
        similarity_df = helper.find_similarity_user_msg(selected_user, df)

        st.title("User Message Similarity")

        fig, ax = plt.subplots()

        ax = sns.heatmap(similarity_df)
        st.pyplot(fig) 



