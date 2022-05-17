from pickle import TRUE
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import folium 
import plotly.express as px
import random
import datetime as dt
import plotly.graph_objs as go
import seaborn as sns
from math import ceil
from PIL import Image
from matplotlib.gridspec import GridSpec
pd.set_option('display.max_columns', 100)
from matplotlib.ticker import FuncFormatter

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="Customer Segmentation Application", page_icon="https://upload.wikimedia.org/wikipedia/en/thumb/4/43/AsiaPacificUniversityOfTechnology%26Innovation.svg/1200px-AsiaPacificUniversityOfTechnology%26Innovation.svg.png", layout="wide")


hide_streamlit_style = """

<style>

.reportview-container .main .block-container{

{

padding-top: {
padding
}rem;

padding-right: {

padding

}rem;

padding-left: {

padding

}rem;

padding-bottom: {

padding

}rem;

}

}

footer {

visibility: hidden;

}

footer:after {

content:'Developed by GOH WEE JER';

visibility: visible;

display: block;

position: relative;

#background-color: red;

padding: 5px;

top: 2px;

}

</style>

"""

st.markdown(
    hide_streamlit_style,
    unsafe_allow_html=True
)

def thousand_count_y(y, pos):
    return '{:.0f} K'.format(y*1e-3)
formatter_thousand_count_y = FuncFormatter(thousand_count_y)


#######################################################
## Overview Page
#######################################################
def overview():  
    
    st.markdown('''
    ## Customer Segmentation Application 👥
    ''')

    st.write('Divides customers into groups based on common characteristics so you can market to each group effectively!')

    with st.expander("RFM Metrics"):
        st.write("""
            RFM stands for `Recency`, `Frequency`, and `Monetary` value, each corresponding to some key customer trait.

            The calculation to know which segmentation the customer is in is given by averaging the Frequency and Monetary scores and the separate Recency value.\n

            | Segment Name               | Range of R values | Range of F and M Average |
            |----------------------------|-------------------|--------------------------|
            | Champions                  |       4 - 5       |           4 - 5          |
            | Loyal Customers            |       2 - 5       |           3 - 5          |
            | Potential Loyalist         |       3 - 5       |           1 - 3          |
            | New Customers              |       4 - 5       |           0 - 1          |
            | Promising                  |       3 - 4       |           0 - 1          |
            | Customers Needing Attetion |       2 - 3       |           2 - 3          |
            | About to Sleep             |       2 - 3       |           0 - 2          |
            | At Risk                    |       0 - 2       |           2 - 5          |
            | Can't Lose Them            |       0 - 1       |           4 - 5          |
            | Hibernating                |       1 - 2       |           1 - 2          |
            | Lost                       |       0 - 2       |           0 - 2          |
        """)

    with st.expander("About Application"):
        st.write("Use this simple app to visualise your dataset, and understand your customers.  It can automate the EDA process and performs customer segmentation using K-means clustering method.")   


#######################################################
## EDA Page
#######################################################
def dataInsight(df):

    st.subheader('Currently selected data:')
    col1, col2= st.columns(2)
    col1.metric("Data Inputs", len(df.index))
    col2.metric("Columns", df.shape[1])

    with st.expander('Click here to see the raw data first 👉'):

        values = st.slider(
            'Select a range of rows',
            5, 100)
    
        st.dataframe(df.head(values))

    
    st.markdown('#### Summary Statistics')
    st.write(df.describe(include = [np.number]).round(1))

    col1, col2 = st.columns(2)
    
    with col1:
        st.write(' ')
        st.write(' ')
        numerical_features = df.select_dtypes(include=['int64','float64', 'uint8']).columns
        st.markdown("#### Numerical Columns")
        st.write(numerical_features)

    with col2:
        st.write(' ')
        st.write(' ')
        categorical_features = df.select_dtypes(exclude=['int64', 'float64', 'uint8']).columns
        st.markdown("#### Categorical Columns")
        st.write(categorical_features)

    st.write(' ')

def dashboard(df):
    fig = plt.figure(constrained_layout=True, figsize=(15, 8))

    # Axis definition
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Lineplot - Evolution of e-commerce orders along time 
    sns.lineplot(x="month_year", y="order_id", data=df.groupby("month_year").agg({"order_id" : "count"}).reset_index(), ax=ax1, legend=False,
                marker='o',markersize=8)
    sns.barplot(x="month_year", y="order_id", data=df.groupby("month_year").agg({"order_id" : "count"}).reset_index(), ax=ax1, alpha=0.1)
    plt.setp(ax1.get_xticklabels(), rotation=45)
    ax1.set_title("Orders in Brazilian e-commerce", size=14, color='black')
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    ax1.set_ylim(0,8500)
    for p in ax1.patches:
            ax1.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points',
                        color= 'black', size=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.set_yticklabels([])
    ax1.set_yticks([])

    #Total Orders
    ax1.text(-1, 7600, "96,708", fontsize=22, ha='center', color="navy")
    ax1.text(-1, 7200, "Total Customers", fontsize=10, ha='center')
    ax1.text(-1, 6000, "42,697", fontsize=18, ha='center', color="navy")
    ax1.text(-1, 5600, "Customers 2017", fontsize=8, ha='center')
    ax1.text(-1, 4400, "54,011", fontsize=18, ha='center', color="navy")
    ax1.text(-1, 4000, "Customers 2018", fontsize=8, ha='center')

    # Barchart - Total of orders by day of week
    day_order= ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df['order_purchase_day'] = df['order_purchase_dayofweek'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
    sns.countplot(x="order_purchase_day", data=df, order=day_order, ax=ax2, palette="YlGnBu")
    ax2.set_title("Orders by Day of Week", size=14, color='black')
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    for p in ax2.patches:
            ax2.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                        ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                        color= "black")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.set_major_formatter(formatter_thousand_count_y)

    # Barchart - Total of orders by time of the day
    day_color_list = ['darkslateblue', 'deepskyblue', 'darkorange', 'purple']
    sns.countplot(x="order_purchase_time_day", data=df,ax=ax3, palette=day_color_list)
    ax3.set_title("Orders by Time of the Day", size=14, color='black')
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    for p in ax3.patches:
            ax3.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                        ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                        color= "black")
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.set_major_formatter(formatter_thousand_count_y)
    
    st.pyplot(plt)
    st.write(' ')
    st.write(' ')
    ##############
    from PIL import Image
    image = Image.open('Unknown.png')
    ##image1 = Image.open('/Users/roger/Downloads/Unknown2.png')
    image3 = Image.open('Unknown3.png')
    image4 = Image.open('Unknown4.png')
    col1, col2 = st.columns((9,1))
    with col1:
        st.image(image4)
    st.write(' ')
    st.write(' ')
    col1, col2 = st.columns((9,1))
    with col1:
        st.image(image)
    st.write(' ')
    st.write(' ')
    col1, col2 = st.columns((9,1))
    with col1:
        st.image(image3)
    
    ###############

def dashboard2(df):
    month_year_line = df['month_year'].value_counts().sort_index()

    fig = px.line(month_year_line, labels = {'index':'Timestamp'}, title = 'Evolution of Total Orders in Brazilian E-Commerce')
    fig.update_layout(
        autosize=False,
        width=1250,
        height=600,
    )
    st.plotly_chart(fig)

    col1, col2 = st.columns(2)

    with col1:
        temp = pd.DataFrame()
        temp['order_purchase_dayofweek'] = df['order_purchase_dayofweek'].sort_values(ascending = True)
        temp['order_purchase_day'] = temp['order_purchase_dayofweek'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})

        fig = px.histogram(temp, x="order_purchase_day", title ='Total orders by Day of Week',text_auto = True)
        st.plotly_chart(fig)

    with col2:
        fig = px.histogram(df, x="order_purchase_time_day", title ='Total orders by Time of the Day',text_auto = True)
        st.plotly_chart(fig)

    col1, col2 = st.columns(2)   

    with col2:
        payment_type = df["payment_type"].nunique()
        customer_payment = df.groupby('payment_type')['customer_id'].nunique().sort_values(ascending = False)

        fig = px.pie(customer_payment, values = customer_payment, names = customer_payment.index, title = 'Proportion of payment methods', hole = .7)
        fig.update_layout(
            annotations=[dict(text='Payment Registered', x=0.5, y=0.5, font_size=15, showarrow=False)]
        )
        st.plotly_chart(fig)

    with col1:
        fig = px.scatter(df, x = 'payment_value', y = 'purchase_delivery_difference', color = 'review_score_category', 
                    labels  = {'payment_value': 'Item Cost', 'purchase_delivery_difference': 'Time taken for delivery'})

        st.plotly_chart(fig)

    with col1:
        customer_city = df.groupby('customer_city',as_index=False)['customer_id'].nunique().sort_values('customer_id', ascending=False).head(10)
        customer_city.columns=['City','Number of Order']
        fig = px.bar(customer_city, y = 'City', x = 'Number of Order', title = 'Top Cities with highest number of Orders', text_auto=True)
        st.plotly_chart(fig)

    with col2: 
        total_customer_state = pd.DataFrame(df['customer_state'].value_counts().reset_index().sort_values(by=['customer_state'],ascending=True))
        total_customer_state.rename(columns = {'index': 'customer_state','customer_state': 'Count'}, inplace = True)
        total_customer_state = total_customer_state[:20]
        fig = px.bar(total_customer_state, y = 'customer_state', x = 'Count', 
                        labels={'Count':'No. of Customers'}, 
                            title = 'Number of Customer in each State', text_auto=True)
        st.plotly_chart(fig)

    payment_evl = df.groupby(by=['month_year', 'payment_type'], as_index=False).count()
    payment_evl = payment_evl.loc[:, ['month_year', 'payment_type', 'order_id']]
    payment_evl = payment_evl.sort_values(by=['month_year', 'order_id'], ascending=[True, False])
    fig = px.line(payment_evl, x='month_year', y='order_id', color = 'payment_type', markers = True)
    fig.update_layout(
        autosize=False,
        width=1250,
        height=600,
    )
    st.plotly_chart(fig)

def eda():
    st.markdown('''
    ### Exploratory Data Analysis 📊 
    ''')
    
    #st.write('Explore your data with the help of summary statistics and graphical representations!')


    df = pd.read_csv('Merged_dataset.zip')

    option = st.selectbox(
        'What would you like to analysis?',
        ('Metadata', 'Dashboard', 'Dashboard 2'))
        
    st.write('---')
    numerical_features = df.select_dtypes(include=['int64','float64', 'uint8']).columns
    categorical_features = df.select_dtypes(exclude=['int64', 'float64', 'uint8']).columns

    if option == 'Metadata':
        dataInsight(df)
    if option == 'Dashboard':
        dashboard(df)
    if option == 'Dashboard 2':
        dashboard2(df)


#######################################################
## EDA Page
#######################################################
def customer_segmentation():
    st.subheader('Customer Segmentation 📈')

    df_user = pd.read_csv('segmentation.csv')

    Segment = df_user.groupby('Segment').size()

    fig = px.pie(Segment, values = Segment, names = Segment.index, title = 'Segment proportion')
    #st.plotly_chart(fig)

    option = st.selectbox(
     'What would you like to analysis?',
     ('Segment', 'Score Segment', 'Overall Score'))

    fig = px.scatter_3d(df_user,x='Recency',y='Frequency',z='Monetary',color = option)
    fig.update_layout(
            autosize=False,
            width=1000,
            height=650,
        )
    st.plotly_chart(fig)

    if option == 'Segment':
        df = df_user[['customer_unique_id', 'Segment']]
    
    if option == 'Score Segment':
        df = df_user[['customer_unique_id', 'Score Segment']]

    if option == 'Overall Score':
        df = df_user[['customer_unique_id', 'Overall Score']]

    else:
        df = df_user[['customer_unique_id', 'Segment']]

    @st.cache
    def convert_df(df_user):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df_user.to_csv().encode('utf-8')

    csv = convert_df(df_user)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='segmentation.csv',
        mime='text/csv',
    )




#######################################################
## SideBar
####################################################### 
with st.sidebar:
    page = st.radio(
        "Navigation 🧭",
        ('Overview', 'Exploratory Data Analysis', 'Customer Segmentation'))

if page == 'Exploratory Data Analysis':
    st.markdown('''
    ### Exploratory Data Analysis 📊 
    ''')
    #st.write('Explore your data with the help of summary statistics and graphical representations!')
        
    df = pd.read_csv('Merged_dataset.zip')

    option = st.selectbox(
        'What would you like to analysis?',
        ('Metadata', 'Dashboard', 'Dashboard 2'))
        
    st.write('---')
    numerical_features = df.select_dtypes(include=['int64','float64', 'uint8']).columns
    categorical_features = df.select_dtypes(exclude=['int64', 'float64', 'uint8']).columns

    if option == 'Metadata':
        dataInsight(df)
    if option == 'Dashboard':
        dashboard(df)
    if option == 'Dashboard 2':
        dashboard2(df)
    
if page == 'Customer Segmentation':
    customer_segmentation()
if page == 'Overview':
    overview()

#######################################################   
