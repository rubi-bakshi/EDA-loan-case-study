#!/usr/bin/env python
# coding: utf-8

# # Credit EDA 

# In[1]:


# Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Reading dataset from local

df=pd.read_csv(r"application_data.csv")


# In[3]:


# Determining the shape of the datset

df.shape


# In[4]:


# Cleaning the missing data

# listing the null values columns having more than 30%

emptycol=df.isnull().sum()
emptycol=emptycol[emptycol.values>(0.3*len(emptycol))]
len(emptycol)


# So, there are 64 columns having null values greater than 30% in the dataset

# In[5]:


# Removing those 64 columns
emptycol = list(emptycol[emptycol.values>=0.3].index)
df.drop(labels=emptycol,axis=1,inplace=True)
print(len(emptycol))


# In[6]:


# Checking the columns having less null percentage

df.isnull().sum()/len(df)*100


# So, 'AMT_ANNUITY' columns is having very few null values rows. Hence let's try to impute the missing values

# Since this column is having an outlier which is very large it will be inappropriate to fill those missing values with mean, 
# Hence Median comes to rescue for this and we will fill those missing banks with median value

# In[7]:


# Filling missing values with median

values=df['AMT_ANNUITY'].median()

df.loc[df['AMT_ANNUITY'].isnull(),'AMT_ANNUITY']=values


# In[8]:


# Searching for the column for null values

df.isnull().sum()


# Now, all columns have been with zero null values

# In[9]:


# Removing rows having null values greater than or equal to 30%

emptyrow=df.isnull().sum(axis=1)
emptyrow=list(emptyrow[emptyrow.values>=0.3*len(df)].index)
df.drop(labels=emptyrow,axis=0,inplace=True)
print(len(emptyrow))


# In[10]:


# We will remove unwanted columns from this dataset

unwanted=['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','FLAG_EMAIL','CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY','DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

df.drop(labels=unwanted,axis=1,inplace=True)


# There are some columns where the value is mentioned as 'XNA' which means 'Not Available'. So we have to find the number of rows and columns and implement suitable techniques on them to fill those missing values or to delete them.

# In[11]:


# let's find these categorical columns having these 'XNA' values
    
# For Gender column

df[df['CODE_GENDER']=='XNA'].shape


# In[12]:


# For Organization column

df[df['ORGANIZATION_TYPE']=='XNA'].shape


# So, there are 4 rows from Gender column and 55374 rows from Organization type column 

# In[13]:


# Describing the Gender column to check the number of females and males

df['CODE_GENDER'].value_counts()


# Since, Female is having the majority and only 4 rows are having NA values, we can update those columns with Gender 'F' as there will be no impact on the dataset.

# In[14]:


# Updating the column 'CODE_GENDER' with "F" for the dataset

df.loc[df['CODE_GENDER']=='XNA','CODE_GENDER']='F'
df['CODE_GENDER'].value_counts()


# In[15]:


# Describing the organization type column

df['ORGANIZATION_TYPE'].describe()


# So, for column 'ORGANIZATION_TYPE', we have total count of 307511 rows of which 55374 rows are having 'XNA' values. Which means 18% of the column is having this values.
# Hence if we drop the rows of total 55374, will not have any major impact on our dataset.

# In[16]:


# Hence, dropping the rows of total 55374 have 'XNA' values in the organization type column

df=df.drop(df.loc[df['ORGANIZATION_TYPE']=='XNA'].index)
df[df['ORGANIZATION_TYPE']=='XNA'].shape


# In[17]:


# Casting all variable into numeric in the dataset

numeric_columns=['TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','REGION_POPULATION_RELATIVE','DAYS_BIRTH',
                'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','HOUR_APPR_PROCESS_START','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']

df[numeric_columns]=df[numeric_columns].apply(pd.to_numeric)
df.head(5)


# ------------------------------------------------

# **Derived Metrics**

# Now, Creating bins for continous variable categories column 'AMT_INCOME_TOTAL' and 'AMT_CREDIT'

# In[18]:


# Creating bins for income amount

bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']

df['AMT_INCOME_RANGE']=pd.cut(df['AMT_INCOME_TOTAL'],bins,labels=slot)


# In[19]:


# Creating bins for Credit amount

bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]
slots = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',
        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',
        '800000-850000','850000-900000','900000 and above']

df['AMT_CREDIT_RANGE']=pd.cut(df['AMT_CREDIT'],bins=bins,labels=slots)


# In[20]:


# Dividing the dataset into two dataset of  target=1(client with payment difficulties) and target=0(all other)

target0_df=df.loc[df["TARGET"]==0]
target1_df=df.loc[df["TARGET"]==1]


# In[21]:


# Calculating Imbalance percentage
    
# Since the majority is target0 and minority is target1

round(len(target0_df)/len(target1_df),2)


# The Imbalance ratio is 10.55

# **Univariate analysis for categories**

# **Now, doing Categorical Univariate Analysis in logarithmic scale for target=0(client with no payment difficulties)**

# In[22]:


# Count plotting in logarithmic scale

def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='magma') 
        
    plt.show()


# In[23]:


# PLotting for income range

uniplot(target0_df,col='AMT_INCOME_RANGE',title='Distribution of income range',hue='CODE_GENDER')


# Points to be concluded from the above graph.
# 
# 1. Female counts are higher than male.
# 2. Income range from 100000 to 200000 is having more number of credits.
# 3. This graph show that females are more than male in having credits for that range.
# 4. Very less count for income range 400000 and above.

# In[24]:


# Plotting for Income type

uniplot(target0_df,col='NAME_INCOME_TYPE',title='Distribution of Income type',hue='CODE_GENDER')


# Points to be concluded from the above graph.
# 
# 1. For income type ‘working’, ’commercial associate’, and ‘State Servant’ the number of credits are higher than others.
# 2. For this Females are having more number of credits than male.
# 3. Less number of credits for income type ‘student’ ,’pensioner’, ‘Businessman’ and ‘Maternity leave’.

# In[25]:


# Plotting for Contract type

uniplot(target0_df,col='NAME_CONTRACT_TYPE',title='Distribution of contract type',hue='CODE_GENDER')


# Points to be concluded from the above graph.
# 
# 1. For contract type ‘cash loans’ is having higher number of credits than ‘Revolving loans’ contract type.
# 2. For this also Female is leading for applying credits.

# In[26]:


# Plotting for Organization type in logarithmic scale

sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30

plt.title("Distribution of Organization type for target - 0")

plt.xticks(rotation=90)
plt.xscale('log')

sns.countplot(data=target0_df,y='ORGANIZATION_TYPE',order=target0_df['ORGANIZATION_TYPE'].value_counts().index,palette='cool')

plt.show()


# Points to be concluded from the above graph.
# 
# 1. Clients which have applied for credits are from most of the organization type ‘Business entity Type 3’ , ‘Self employed’,
#    ‘Other’ , ‘Medicine’ and ‘Government’.
# 2. Less clients are from Industry type 8,type 6, type 10, religion and  trade type 5, type 4.
# 

# **Now, doing Categoroical Univariate Analysis in logarithmic scale for target=1(client with payment difficulties)**

# In[27]:


# PLotting for income range

uniplot(target1_df,col='AMT_INCOME_RANGE',title='Distribution of income range',hue='CODE_GENDER')


# Points to be concluded from the above graph.
# 
# 1. Male counts are higher than female.
# 2. Income range from 100000 to 200000 is having more number of credits.
# 3. This graph show that males are more than female in having credits for that range.
# 4. Very less count for income range 400000 and above.
# 

# In[28]:


# Plotting for Income type

uniplot(target1_df,col='NAME_INCOME_TYPE',title='Distribution of Income type',hue='CODE_GENDER')


# Points to be concluded from the above graph.
# 
# 1. For income type ‘working’, ’commercial associate’, and ‘State Servant’ the number of credits are higher than other i.e.   ‘Maternity leave.
# 2. For this Females are having more number of credits than male.
# 3. Less number of credits for income type  ‘Maternity leave’.
# 4. For type 1: There is no income type for ‘student’ , ’pensioner’ and ‘Businessman’ which means they don’t do any late payments.

# In[29]:


# Plotting for Contract type

uniplot(target1_df,col='NAME_CONTRACT_TYPE',title='Distribution of contract type',hue='CODE_GENDER')


# Points to be concluded from the above graph.
# 
# 1. For contract type ‘cash loans’ is having higher number of credits than ‘Revolving loans’ contract type.
# 2. For this also Female is leading for applying credits.
# 3. For type 1 : there is only Female Revolving loans.

# In[30]:


# Plotting for Organization type

sns.set_style('whitegrid')
sns.set_context('talk')
plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30

plt.title("Distribution of Organization type for target - 1")

plt.xticks(rotation=90)
plt.xscale('log')

sns.countplot(data=target0_df,y='ORGANIZATION_TYPE',order=target0_df['ORGANIZATION_TYPE'].value_counts().index,palette='cool')

plt.show()


# Points to be concluded from the above graph.
# 
# 1. Clients which have applied for credits are from most of the organization type ‘Business entity Type 3’ , ‘Self employed’ , ‘Other’ , ‘Medicine’ and ‘Government’.
# 2. Less clients are from Industry type 8,type 6, type 10, religion and  trade type 5, type 4.
# 3. Same as type 0 in distribution of organization type.
# 

# In[31]:


# Finding some correlation for numerical columns for both target 0 and 1 

target0_corr=target0_df.iloc[0:,2:]
target1_corr=target1_df.iloc[0:,2:]

target0=target0_corr.corr(method='spearman')
target1=target1_corr.corr(method='spearman')


# In[32]:


# Correlation for target 0

target0


# In[33]:


# Correlation for target 1

target1


# In[34]:


# Now, plotting the above correlation with heat map as it is the best choice to visulaize

# figure size

def targets_corr(data,title):
    plt.figure(figsize=(15, 10))
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['axes.titlepad'] = 70

# heatmap with a color map of choice


    sns.heatmap(data, cmap="RdYlGn",annot=False)

    plt.title(title)
    plt.yticks(rotation=0)
    plt.show()


# In[35]:


# For Target 0

targets_corr(data=target0,title='Correlation for target 0')


# As we can see from above correlation heatmap, There are number of observation we can point out
# 
# 1. Credit amount is inversely proportional to the date of birth, which means Credit amount is higher for low age and vice-versa.
# 2. Credit amount is inversely proportional to the number of children client have, means Credit amount is higher for less children count client have and vice-versa.
# 3. Income amount is inversely proportional to the number of children client have, means more income for less children client have and vice-versa.
# 4. less children client have in densely populated area.
# 5. Credit amount is higher to densely populated area.
# 6. The income is also higher in densely populated area.

# In[36]:


# For Target 1

targets_corr(data=target1,title='Correlation for target 1')


# This heat map for Target 1 is also having quite a same observation just like Target 0. But for few points are different. They are listed below.
# 
# 1. The client's permanent address does not match contact address are having less children and vice-versa
# 2. the client's permanent address does not match work address are having less children and vice-versa

# **Univariate analysis for variables**

# In[37]:


# Box plotting for univariate variables analysis in logarithmic scale

def univariate_numerical(data,col,title):
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    plt.title(title)
    plt.yscale('log')
    sns.boxplot(data =target1_df, x=col,orient='v')
    plt.show()
    


# **For Target 0 - Finding any outliers** 

# In[38]:


# Distribution of income amount

univariate_numerical(data=target0_df,col='AMT_INCOME_TOTAL',title='Distribution of income amount')


# Few points can be concluded from the graph above.
# 
# 1. Some outliers are noticed in income amount.
# 2. The third quartiles is very slim for income amount.
# 

# In[39]:


# Disrtibution of credit amount

univariate_numerical(data=target0_df,col='AMT_CREDIT',title='Distribution of credit amount')


# Few points can be concluded from the graph above.
# 
# 1. Some outliers are noticed in credit amount.
# 2. The first quartile is bigger than third quartile for credit amount which means most of the credits of clients are present in the first quartile.
# 

# In[40]:


# Distribution of anuuity amount

univariate_numerical(data=target0_df,col='AMT_ANNUITY',title='Distribution of Annuity amount')


# Few points can be concluded from the graph above.
# 
# 1. Some outliers are noticed in annuity amount.
# 2. The first quartile is bigger than third quartile for annuity amount which means most of the annuity clients are from first quartile.
# 

# **For Target 1 - Finding any outliers**

# In[41]:


# Distribution of income amount

univariate_numerical(data=target1_df,col='AMT_INCOME_TOTAL',title='Distribution of income amount')


# Few points can be concluded from the graph above.
# 
# 1. Some outliers are noticed in income amount.
# 2. The third quartiles is very slim for income amount.
# 3. Most of the clients of income are present in first quartile.
# 

# In[42]:


# Distribution of credit amount

univariate_numerical(data=target1_df,col='AMT_CREDIT',title='Distribution of credit amount')


# Few points can be concluded from the graph above.
# 
# 1. Some outliers are noticed in credit amount.
# 2. The first quartile is bigger than third quartile for credit amount which means most of the credits of clients are present in the first quartile.

# In[43]:


# Distribution of Annuity amount

univariate_numerical(data=target1_df,col='AMT_ANNUITY',title='Distribution of Annuity amount')


# Few points can be concluded from the graph above.
# 
# 1. Some outliers are noticed in annuity amount.
# 2. The first quartile is bigger than third quartile for annuity amount which means most of the annuity clients are from first quartile.

# **Bivariate analysis for numerical variables**

# **For Target 0**

# In[44]:


# Box plotting for Credit amount

plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
sns.boxplot(data =target0_df, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Credit amount vs Education Status')
plt.show()


# From the above box plot we can conclude that Family status of 'civil marriage', 'marriage' and 'separated' of Academic degree education are having higher number of credits than others.
# Also, higher education of family status of 'marriage', 'single' and 'civil marriage' are having more outliers.
# Civil marriage for Academic degree is having most of the credits in the third quartile.

# In[45]:


# Box plotting for Income amount in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
plt.yscale('log')
sns.boxplot(data =target0_df, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Income amount vs Education Status')
plt.show()


# From above boxplot for Education type 'Higher education' the income amount is mostly equal with family status. It does contain many outliers. Less outlier are having for Academic degree but there income amount is little higher that Higher education.
# Lower secondary of civil marriage family status are have less income amount than others.

# **For Target 1**

# In[46]:


# Box plotting for credit amount

plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
sns.boxplot(data =target0_df, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Credit Amount vs Education Status')
plt.show()


# Quite similar with Target 0
# From the above box plot we can say that Family status of 'civil marriage', 'marriage' and 'separated' of Academic degree education are having higher number of credits than others. Most of the outliers are from Education type 'Higher education' and 'Secondary'.
# Civil marriage for Academic degree is having most of the credits in the third quartile.

# In[47]:


# Box plotting for Income amount in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
plt.yscale('log')
sns.boxplot(data =target0_df, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v')
plt.title('Income amount vs Education Status')
plt.show()


# Have some similarity with Target0,
# From above boxplot for Education type 'Higher education' the income amount is mostly equal with family status. Less outlier are having for Academic degree but there income amount is little higher that Higher education.
# Lower secondary are have less income amount than others.

# In[48]:


# Reading the dataset of previous application

df1=pd.read_csv(r"previous_application.csv")


# In[49]:


# Cleaning the missing data

# listing the null values columns having more than 30%

emptycol1=df1.isnull().sum()
emptycol1=emptycol1[emptycol1.values>(0.3*len(emptycol1))]
len(emptycol1)


# In[50]:


# Removing those 15 columns

emptycol1 = list(emptycol1[emptycol1.values>=0.3].index)
df1.drop(labels=emptycol1,axis=1,inplace=True)

df1.shape


# In[51]:


# Removing the column values of 'XNA' and 'XAP'

df1=df1.drop(df1[df1['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)
df1=df1.drop(df1[df1['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)
df1=df1.drop(df1[df1['NAME_CASH_LOAN_PURPOSE']=='XAP'].index)

df1.shape


# In[52]:


# Now merging the Application dataset with previous appliaction dataset

new_df=pd.merge(left=df,right=df1,how='inner',on='SK_ID_CURR',suffixes='_x')


# In[53]:


# Renaming the column names after merging

new_df1 = new_df.rename({'NAME_CONTRACT_TYPE_' : 'NAME_CONTRACT_TYPE','AMT_CREDIT_':'AMT_CREDIT','AMT_ANNUITY_':'AMT_ANNUITY',
                         'WEEKDAY_APPR_PROCESS_START_' : 'WEEKDAY_APPR_PROCESS_START',
                         'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START','NAME_CONTRACT_TYPEx':'NAME_CONTRACT_TYPE_PREV',
                         'AMT_CREDITx':'AMT_CREDIT_PREV','AMT_ANNUITYx':'AMT_ANNUITY_PREV',
                         'WEEKDAY_APPR_PROCESS_STARTx':'WEEKDAY_APPR_PROCESS_START_PREV',
                         'HOUR_APPR_PROCESS_STARTx':'HOUR_APPR_PROCESS_START_PREV'}, axis=1)


# In[54]:


# Removing unwanted columns for analysis

new_df1.drop(['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION', 
              'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
              'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY','WEEKDAY_APPR_PROCESS_START_PREV',
              'HOUR_APPR_PROCESS_START_PREV', 'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)


# **Performing univariate analysis**

# In[55]:


# Distribution of contract status in logarithmic scale

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of contract status with purposes')
ax = sns.countplot(data = new_df1, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=new_df1['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS',palette='magma') 


# Points to be concluded from above plot:
# 
# 1. Most rejection of loans came from purpose 'repairs'.
# 2. For education purposes we have equal number of approves and rejection
# 3. Payign other loans and buying a new car is having significant higher rejection than approves.

# In[56]:


# Distribution of contract status

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of purposes with target ')
ax = sns.countplot(data = new_df1, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=new_df1['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'TARGET',palette='magma') 


# Few points we can conclude from abpve plot:
# 
# 1. Loan purposes with 'Repairs' are facing more difficulites in payment on time.
# 2. There are few places where loan payment is significant higher than facing difficulties.
#     They are 'Buying a garage', 'Business developemt', 'Buying land','Buying a new car' and 'Education'
# Hence we can focus on these purposes for which the client is having for minimal payment difficulties.

# **Performing bivariate analysis**

# In[57]:


# Box plotting for Credit amount in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
plt.yscale('log')
sns.boxplot(data =new_df1, x='NAME_CASH_LOAN_PURPOSE',hue='NAME_INCOME_TYPE',y='AMT_CREDIT_PREV',orient='v')
plt.title('Prev Credit amount vs Loan Purpose')
plt.show()


# From the above we can conclude some points-
# 
# 1. The credit amount of Loan purposes like 'Buying a home','Buying a land','Buying a new car' and'Building a house' is higher.
# 2. Income type of state servants have a significant amount of credit applied
# 3. Money for third person or a Hobby is having less credits applied for.

# In[58]:


# Box plotting for Credit amount prev vs Housing type in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
sns.barplot(data =new_df1, y='AMT_CREDIT_PREV',hue='TARGET',x='NAME_HOUSING_TYPE')
plt.title('Prev Credit amount vs Housing type')
plt.show()


# Here for Housing type, office appartment is having higher credit of target 0 and co-op apartment is having higher credit of target 1.
# So, we can conclude that bank should avoid giving loans to the housing type of co-op apartment as they are having difficulties in payment.
# Bank can focus mostly on housing type with parents or House\appartment or miuncipal appartment for successful payments.

# # CONCLUSION

# **1. Banks should focus more on contract type ‘Student’ ,’pensioner’ and ‘Businessman’ with housing ‘type other
#      than ‘Co-op apartment’ for successful payments.**
# 
# **2. Banks should focus less on income type ‘Working’ as they are having most number of unsuccessful payments.**
# 
# **3. Also with loan purpose ‘Repair’ is having higher number of unsuccessful payments on time.**
# 
# **4. Get as much as clients from housing type ‘With parents’ as they are having least number of unsuccessful payments.**
# 
