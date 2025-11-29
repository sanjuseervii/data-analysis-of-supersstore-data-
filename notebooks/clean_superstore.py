import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
data=pd.read_csv("data//superstore.csv")
df=pd.DataFrame(data)
'''print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
#data type transferring
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date']  = pd.to_datetime(df['Ship Date'], dayfirst=True)
df['year_od']=df['Order Date'].dt.year
df['month_od']=df['Order Date'].dt.month
df['days']=df['Order Date'].dt.day
df['shipping days']=(df['Ship Date']-df['Order Date']).dt.days
#handling missing values 
df['Postal Code']=df['Postal Code'].fillna(df['Postal Code'].median(),inplace=True)
sns.boxplot(df['Sales'])
plt.show()
#barplot cataogy and sales 
sns.barplot(x='Category', y='Sales',data=df, estimator=sum)
plt.title("Total Sales by Category")
plt.show()
#sub catagory wise sales 
plt.figure(figsize=(10,6))
sns.barplot(x='Sub-Category',y='Sales',data=df,estimator=sum)
plt.xticks(rotation=45)
plt.title("category wise sales")
plt.show()
#region wise sales 
sns.barplot(x='Region',y="Sales",data=df)
plt.title('Region Wise Sales')
plt.show()
corr=df.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title("correlation heatmap ")
plt.show()
#top 10 customer
top_customer=df.groupby("Customer Name")['Sales'].sum().sort_values(ascending=False).head(10)
print('top 10 customer/n',top_customer)
#top 10 city 
top_city=df.groupby("City")['Sales'].sum().sort_values(ascending=False).head(10)
print("top 10 cities/n",top_city)
#top ten product 
top_product=df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
print('top_product/n',top_product)
print(df.info())
#adding a new column just for demostration purpose
#Profit = Sales * random margin (5%–20%)
def calculate_profit(row):
    if row['Category'] == 'Technology':
        margin = np.random.uniform(0.15, 0.25)
    elif row['Category'] == 'Furniture':
        margin = np.random.uniform(0.10, 0.18)
    else:  # Office Supplies
        margin = np.random.uniform(0.05, 0.12)
    
    return row['Sales'] * margin

# Apply the function
df['Profit'] = df.apply(calculate_profit, axis=1)

df['Profit'] = df['Profit'].round(2)

print(df.head())
#category wise profit
top_cat_profit=df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
#product wise profit
product_profit=df.groupby("Product Name")["Profit"].sum().sort_values(ascending=False).head(10)
#city wise profit
city_profit=df.groupby("City")["Profit"].sum().sort_values(ascending=False).head(10)
print("category wise profit")
print(top_cat_profit)
print('product wise profit')
print(product_profit)
print('city wise profit')
print(city_profit)
#loss flag 
df['Loss_Flag'] = 0  # default no loss

# Condition 1: Long shipping times → higher loss probability
df.loc[df['shipping days'] > 6, 'Loss_Flag'] = 1

# Condition 2: Furniture with very low sales → loss
df.loc[(df['Category'] == 'Furniture') & (df['Sales'] < 100), 'Loss_Flag'] = 1

# Condition 3: Random 5% orders marked as loss (realistic business noise)
random_loss_index = df.sample(frac=0.05, random_state=42).index
df.loc[random_loss_index, 'Loss_Flag'] = 1
#category wise analysis
total_loss=df['Loss_Flag'].sum()
mean_loss=(df['Loss_Flag'].mean() * 100).round(2)
category_loss=df.groupby("Category")["Loss_Flag"].sum()
city_loss=df.groupby("City")["Loss_Flag"].sum().sort_values(ascending=False).head(10)
print('total loss:',total_loss)
print('mean loss',mean_loss)
print('city loss')
print(city_loss)
#profit loss visualization
plt.figure(figsize=(7,5))

sns.barplot(x="Category", y="Loss_Flag", data=df, estimator=sum)
plt.title("Total Loss Orders by Category")
plt.show()

loss_cities = df.groupby("City")["Loss_Flag"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=loss_cities.index, y=loss_cities.values)
plt.xticks(rotation=45)
plt.title("Top 10 Loss-Making Cities")
plt.show()
plt.figure(figsize=(7,5))
sns.scatterplot(x="Sales", y="Profit", hue="Category", data=df)
plt.title("Profit vs Sales (Colored by Category)")
plt.show()
latest_date = df['Order Date'].max()
rfm=df.groupby('Customer Name').agg({'Order Date':lambda x:
                                     (latest_date-x.max()).days,
                                     'Order ID':'count',
                                     'Sales':'sum'})
rfm.rename(columns={
            'Order Date':'recency',
            "Order ID":'frequency',
            'Sales':'Monetary'
},inplace=True)
rfm['R_score']=pd.qcut(rfm['recency'],4,labels=[4,3,2,1])#recency low days=best,best=4,worst=1
rfm['F_score']=pd.qcut(rfm['frequency'].rank(method='first'),4,labels=[1,2,3,4])#frequency high=best customer
rfm['M_score']=pd.qcut(rfm['Monetary'],4,labels=[1,2,3,4])#money high best score 4 
rfm['rfm_score']=rfm['R_score'].astype(str)+rfm['F_score'].astype(str)+rfm['M_score'].astype(str)
top_rfm=rfm.sort_values(by='rfm_score',ascending=False).head(10)
print('TOP RFM CUSTOMERS')
print(top_rfm)
def segment(row):
    if row.R_score==4 and row.F_score==4 and row.M_score==4:
        return 'Champions'
    elif row.R_score>=3 and row.F_score>=3:
        return 'loyal customer'
    elif row.R_score==4  and row.F_score<=2:
        return 'potenial loyalist'
    elif row.R_score==2 and row.F_score>=2:
        return 'needs attention'
    elif row.R_score==1 and row.F_score>=2:
        return 'at risk'
    else:
        return 'Lost'
rfm['rfm_segment']=rfm.apply(segment,axis=1)
#final rfm table
print(rfm.head(20))
#rfm visualization 
plt.figure(figsize=(10,5))
rfm['rfm_segment'].value_counts().plot(kind='bar')
plt.title ('customer segments count')
plt.xlabel('segment')
plt.ylabel('count')
plt.xticks(rotation=45)
plt.show()
#merge rfm score with original df
df.columns=df.columns.str.strip()
rfm.columns=rfm.columns.str.strip()
rfm=rfm.reset_index()
dff=df.merge(rfm[['Customer Name','rfm_segment']],on='Customer Name',how='left')
print("merge succes")
print("SEGMENTATION CUSTOMER COUNT")
#SEGMENTATION VS AVERAGE sales 
print('segmentation average')
plt.figure(figsize=(10,5))
dff.groupby('rfm_segment')['Sales'].mean().sort_values().plot(kind='bar')
plt.title("average sales by segment")
plt.xlabel("Segment")
plt.ylabel('average sales')
plt.xticks(rotation=45)
plt.show()
#region wise segment distribution
plt.figure(figsize=(12,6))
region_seg=dff.groupby(['Region','rfm_segment']).size().unstack()
region_seg.plot(kind='bar')
plt.title("region wise segment distribution")
plt.xlabel('Region')
plt.ylabel('count')
plt.xticks(rotation=45)
plt.show()
#category wise rfm segmant count
plt.figure(figsize=(12,6))
cat_seg=dff.groupby(['Category','rfm_segment']).size().unstack()
cat_seg.plot(kind='bar')
plt.title('category wise segment distribution')
plt.xlabel('category')
plt.ylabel('count')
plt.xticks(rotation=50)
plt.show()
#top 10 customers by monetary
plt.figure(figsize=(12,6))
top_cust=dff.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10)
top_cust.plot(kind='bar')
plt.title('top 10 customers by total sales ')
plt.ylabel('sales')
plt.xlabel('customer')
plt.xticks(rotation=45)
plt.show()
#heatmap(segment vs category)
pivot=dff.pivot_table(index='rfm_segment',columns='Category',values='Sales',aggfunc='mean')
plt.figure(figsize=(12,6))
sns.heatmap(pivot,annot=True,fmt='.1f')
plt.title('avg sales heatmap (segment vs category)')
plt.show()
#pairplot(rfm)
sns.pairplot(rfm[['recency','frequency','Monetary']])
plt.show()'''
#fm.to_csv('rfm_model.csv',index=False)
print(df.columns)