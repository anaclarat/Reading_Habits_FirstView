# -*- coding: utf-8 -*-
"""
Created on Sat Jul 05 18:01:11 2021

@author: Ana Clara Tupinambá Freitas

"""
# =============================================================================
# References:
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.skew.html
#https://vknight.org/unpeudemath/python/2016/08/13/Analysis-of-variance-with-different-sized-sample.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html#scipy.stats.kruskal
#https://influentialpoints.com/Training/kruskal-wallis_anova-principles-properties-assumptions.htm
#https://www.real-statistics.com/one-way-analysis-of-variance-anova/kruskal-wallis-test/
#https://data.library.virginia.edu/understanding-q-q-plots/
#https://www.researchgate.net/post/Kolmogorov-Smirnov_test_or_Shapiro-Wilk_test_which_is_more_preferred_for_normality_of_data_according_to_sample_size
#https://stats.stackexchange.com/questions/396717/qq-plot-and-shapiro-wilk-test-disagree/396753
#https://stats.stackexchange.com/questions/52293/r-qqplot-how-to-see-whether-data-are-normally-distributed
#https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/

# Data set source:
# https://www.kaggle.com/vipulgote4/reading-habit-dataset
# =============================================================================

# =============================================================================
# Libraries/Packages
# =============================================================================
#Install new package
#!pip install researchpy 

# Importing Packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from seaborn_qqplot import pplot
import researchpy as rp
from statsmodels.graphics.gofplots import qqplot
#from scipy.stats import shapiro
from scipy.stats import chi2_contingency
#from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import levene
from scipy.stats import f_oneway
from scipy.stats import kruskal
from scipy.stats import kstest

# =============================================================================
# Directory
# =============================================================================
os.chdir(r'D:\Repositories\Reading_Habits_FirstView')
os.getcwd()

# =============================================================================
# UDFs (not prorammed to receive null values,yet)
# =============================================================================
plt.interactive(False) #graph will only be showed when plt.show() is called

def uni_numerical(df):
    '''This function returns summarization and visual for univariate analysis of numerical features of a dataset'''
    
# Summarization
    print('############### Univariate analysis - Numerical ###############')
    df.describe()

    Summary = dict(df.describe())
    for i in Summary:
        Summary[i]['count'] = df[i].count()
        Summary[i]['variance'] = df[i].var()
        Summary[i]['IQR'] = Summary[i]['75%'] - Summary[i]['25%']
        Summary[i]['range'] = Summary[i]['max'] - Summary[i]['min']
        Summary[i]['skewness'] = df[i].skew()
        Summary[i]['kurtosis'] = df[i].kurtosis()
        Summary[i]['mode'] = df[i].mode() #axis='columns'
#    print(Summary)
        print('\nThis is univariate analysis for',"'", i, "'", '\n', Summary[i])
# Visualization
        # Histogram and Boxplot
        sns.set_palette("dark") # colorblind   pastel
        sns.set(style="white") #whitegrid ticks
        sns.set_context("paper", font_scale=1.5)
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.12, .88)})
        f.set_size_inches(12, 8)
        sns.distplot(a=df[i], hist=True, kde=True, rug=True, ax=ax_hist)
        sns.boxplot(x=df[i], ax=ax_box)
        ax_box.set_title('Univariate plot')
        ax_box.set(xlabel='')
        plt.show()
        #QQ-plot
#        plt.show()
        with plt.rc_context():
            plt.rc("figure", figsize=(12, 8))
            qqplot(df[i], line='s') #standardized line, the expected order statistics are scaled by the standard deviation of the given sample and have the mean added to them
            plt.title('Univariate plot: '+ i)
            plt.show()
# normality test
#        stat, p_normal = shapiro(df[i])
        stat, p_normal = kstest(df[i],'norm')    
#        print('Shapiro\'s Statistics=%.3f, p=%.3f' % (stat, p_normal))
        print('Kolmogorov\'s Statistics=%.3f, p=%.3f' % (stat, p_normal))
# interpretation
        alpha = 0.05
        if p_normal > alpha:
            print('NNormality\'s test result: Sample looks Gaussian (fail to reject H0)')
        else:
            print('Normality\'s test result: Sample does not look Gaussian (reject H0)')
#uni_numerical(df1)

#-----------------------------------------

def uni_categorical(df):
    '''This function returns summarization and visual for univariate analysis of numerical features of a dataset'''
    
    print('############### Univariate analysis - Categorical ###############')
    df = df.select_dtypes(exclude=np.number) #extracting only non-numerical features
    for i in df.columns:
        if type(df.loc[0, i]) == str: #checking type of features
# Summarization
            d1 = df[i].value_counts()
            d2 = round(df[i].value_counts(normalize=True)*100, 2)
            d3 = pd.concat([d1, d2], axis=1)
            d3.columns = ['Count', 'Percentage']
            print('\nThis is univariate analysis for', "'", i, "'", '\n', d3)

 # Visualization
            d1 = pd.DataFrame(d1).T
            d1 = pd.melt(d1,var_name=str(i),value_vars=d1) 
            fig, ax = plt.subplots()
            fig.set_size_inches(12, 8)
            sns.set_context("paper", font_scale=1.5)
            plot1=sns.barplot(y=d1['value'],x=d1[i], data=d1)
            for p, label in zip(plot1.patches, d1['value']):
                ax.annotate(label, (p.get_x()+0.3, p.get_height()+2))
            plt.xticks(rotation=45, ha='right')
            ax.set_title('Univariate plot')
            plt.show()
    print('\n\n')
#uni_categorical(df2)

# --------------------------------------------

def bi_numerical_num(df, var_target):
    '''This function returns summarization, visual,and test of independency for bivariate analysis of numerical x numerical features. '''
    
    print('############### Bivariate analysis - Numerical x Numerical ###############')
# Printing Assumptions:
    print('\nCorrelation (how strong the correlation is):\n Null hypothesis: there’s no association between variables.\n \t1.Normal distribution for both variables for pearson;\n  \t2.homoscedasticity assumes that data is equally distributed about the regression line.\n \t3.Linear? \n \t\tLinear: pearson\n \t\tMonotonically related (not normal): spearman kendall hoeffding \n')

# Verifying datatypes extract only numerics
    df = df.select_dtypes(include=np.number)
#numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# verifying normality
#    uni_numerical(df)

# correlation test
    print('\nThis is Pearson\'s correlation:')
    print(df.corrwith(df[var_target], method='pearson'), '\n')
    print('This is Spearman\'s correlation:')
    print(df.corrwith(df[var_target], method='spearman'), '\n')
# Visualization
    sns.set_palette("dark") # colorblind   pastel
    sns.set(style="white") #whitegrid ticks
    fig=sns.jointplot(*df,kind='reg',data=df,joint_kws={'line_kws':{'color':'red',}},scatter_kws= {'alpha': 0.4,'s':7})
    fig.fig.set_size_inches(12, 8)
    plt.show()
    print('\n\n')
#    fig =sns.pairplot(df, kind='reg',diag_kind='hist',height=4,diag_kws=dict(alpha=0.4),plot_kws={'scatter_kws': {'alpha': 0.4,'s':5},
#            'line_kws':{'color':'red'}})#,corner=True not working
#    fig.fig.set_size_inches(12, 8)
#    plt.show()
    
#bi_numerical_num(df2,'How many books did you read during last 12months?')

# --------------------------------------------

def bi_categorical_cat(df, var_target):
    '''This function returns summarization, visual, and test of independency for bivariate analysis of categorical x categorical features. '''
    
    print('############### Bivariate analysis - Categorical x Categorical ###############')
# Printing Assumptions:
    print('\nChi – square:\n If condition of chi-square are satisfied and p-value is less than significant level (5%) reject null hypothesis: There is a relationship between variables at 5% significant level.\n Null hypothesis: Variables are independents.\n 1. N, the total frequency, should be reasonably large (greater than 50);\n 2. The sample observations should be independent. No individual item should be included twice or more in the sample;\n 3. No expected frequencies should be small. Preferably each expected frequency should be larger than 10 but in any case not less than 5.\n')
    df = df.select_dtypes(exclude=np.number) #extracting only non-numerical features
    for i in df.columns:
        if i != var_target:           
            tab = pd.crosstab(df[i], df[var_target], margins = True)
            print('\n\nThis is bivariate analysis for ',i,' and ',var_target)
            print('\n', pd.crosstab(df[i], df[var_target], margins = True))
            pd.crosstab(df[var_target],df[i], margins = False).plot(kind='bar',stacked=False, figsize=(12, 8))#yerr=[df[var_target].mean()-df[var_target].min(), df[var_target].max()-df[var_target].mean()])
            plt.xticks(rotation=45, ha='right')
            plt.title('Bivariate plot: ' + var_target + ' and ' + i)
            plt.show()

#            -------------------------------------------
        # independency test
            x2_value, p, degf, exp_val = chi2_contingency(tab)
            print('p-value is: p=%.3f' % (p))
        # interpretation
            alpha = 0.05
            if p > alpha:
                print('Chi-square\'s test result: Features are independent (fail to reject H0)')
            else:
                print('Chi-square\'s test result: Features are dependent (reject H0)')   
    print('\n\n')
#bi_categorical_cat(df2,'Number of Books (Segmented)')        

# --------------------------------------------
def bi_categorical_cont(df,var_target):                
    '''This function returns summarization, visual, and test of independency for bivariate analysis of categorical x continuous features,taking a continuous variable as the target.  '''
    
    print('############### Bivariate analysis - Categorical x Numerical ###############')
    print('\n*If Categorical has 2 levels: Mann-Whitney U test, if more: One-way ANOVA/Kruskal H test')

# Verifying datatypes extract only numerics
    df_num = df.select_dtypes(include=np.number) 
    
# If target is numeric:         
    if var_target in df_num.columns:
        df_cat = df.select_dtypes(exclude=np.number)
        for i in df_cat.columns:
            if i != var_target:
                vis = pd.crosstab(df[var_target],df[i], margins = False)
                tab = rp.summary_cont(df[var_target].groupby(df[i]),decimals=3)#conf=0.95
#                print(tab)
                result = df.groupby(i)[var_target].apply(list) 
#                print('Result:',result)
# Printing Assumptions:
                if len(result.index) == 2:
                    print('\nThis is Mann-Whitney U test for' , var_target, 'and',i )
                    print('\nMann-Whitney U test:\n Null hypothesis: The distribution of scores for the two groups are equal.\n Assumptions:\n 1.Dependent variable that is measured at the continuous or ordinal level.\n 2.Groups are independent of one another. \n 3.What\'s the shape of each group distribution.\n\t a) Similar shapes: \n\t\t Alternative hypothesis: the medians of the two groups are not equal.\n\t b) Different shapes:\n\t\t Alternative hypothesis: the distribution of scores for the two groups are not equal.') 
                    print('\nSummary:')
                    print(tab)
# Visualization
                    sns.set_palette("dark") # colorblind   pastel
                    sns.set(style="white") # whitegrid ticks
                    fig, ax = plt.subplots()
                    fig.set_size_inches(12, 8)
                    sns.set_context("paper", font_scale=1.5)
                    l = pd.melt(vis,var_name=str(i),value_vars=vis)
                    l = pd.DataFrame(l)
                    sns.boxplot(y=l['value'],x=l[i], data=l)  
                    sns.stripplot(x=l['value'], y=l[i], data=l,
              size=4, color=".3", linewidth=0,jitter=0.1)
                    plt.xticks(rotation=45, ha='right')
                    ax.set_title('Bivariate plot: ' + var_target + ' and ' + i)
                    plt.show()
                
                else:
# ANOVA
# Printing Assumptions: 
                    print('\nThis is ANOVA/Kruskal for' , var_target, 'and',i,'\n' )
                    print('One-way ANOVA Assumptions\n In order to run a one-way ANOVA the following assumptions must be met:\n 1.The response of interest is continuous and normally distributed for each treatment group.\n \tCLT :\n \t\tIf looks normal each group must have more than 30 observations – no need for Normality’s test;\n \t\tIf moderately skewed(|0.5| < skewness < |1.0|), each group must have more than 100 observations – no need for Normality’s test;\n\t*If not normal, proceed to Kruskal test.\n 2.Treatment groups are independent of one another. \n 3.There are no major outliers.\n 4.A check for unequal variances will help determine which version of a one-way ANOVA is most appropriate (Levene’s test, Null hypothesis: variances are equal between groups):\n \tA .If variances are equal, then the assumptions of a standard one-way ANOVA are met.\n \tB. If variances are unequal, then a Kruskal’s test is appropriate.\n\n*Kruskal\'s Null hypothesis: H0: the group populations have equal dominance; i.e. when one element is drawn at random from each group population, the largest (or smallest, or second smallest, etc.) element is equally likely to come from any one of the group populations or H0: the group population medians are equal(if groups distributions have the same shape) ')
                    print('\nSummary:')
                    print(tab)
# Visualization
                    sns.set_palette("dark") # colorblind   pastel
                    sns.set(style="white") # whitegrid ticks
                    fig, ax = plt.subplots()
                    fig.set_size_inches(12, 8)
                    sns.set_context("paper", font_scale=1.5)
                    l = pd.melt(vis,var_name=str(i),value_vars=vis)
                    l = pd.DataFrame(l)
                    sns.boxplot(y=l['value'],x=l[i], data=l)  
                    sns.stripplot(y=l['value'], x=l[i], data=l,
              size=4, color=".3", linewidth=0)
                    plt.xticks(rotation=45, ha='right')
                    ax.set_title('Bivariate plot: ' + var_target + ' and ' + i)
                    plt.show()
# Normality test and visual

            for x in result.index:
##--------------------------------
                #QQ-plot
                vis1=l.loc[l[i]== x]
                with plt.rc_context():
                    plt.rc("figure", figsize=(12, 8))
                    qqplot(vis1['value'], line='s') #standardized line, the expected order statistics are scaled by the standard deviation of the given sample and have the mean added to them
                    plt.title('QQ-plot for: '+ str(x))
                    plt.show()
##--------------------------------
#The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.                    
#                print(result[x])    
#                stat_s, p_normal = shapiro(result[x])
                stat_s,p_normal = kstest(result[x], 'norm')
#                print('\nShapiro\'s Statistics for ', x,'=%.3f, p=%.3f' % (stat_s, p_normal))
                skew1 = pd.DataFrame(result[x]).skew()
                print('\nSkewness for ', x,'=%.3f' % (skew1))
                print('\nKolmogorov\'s Statistics for ', x,'=%.3f, p=%.3f' % (stat_s, p_normal))
                alpha = 0.05
                if p_normal  > alpha:     # Gaussian dist   
                    print('Normality\'s test result: Sample looks Gaussian (reject H0)')     

                else: # Non-Gaussian dist
                    print('Normality\'s test result: Sample does not look Gaussian (reject H0)')
                    print('Please refer to test\'s assumptions, before continuing.')

            if len(result.index) == 2:
#T-test                             
                stat_l, p_lev = levene(*result)
                print('\nLevene\'s Statistics=%.3f, p=%.3f' % (stat_l, p_lev))
                if p_lev  > alpha:#Equal varinaces
                    stat_u, p_u = mannwhitneyu(*result)
                    print('Levene\'s test result: Equal variances (fail to reject H0)')
                    print('\nMann-Whitney U \'s Null hypothesis: The distribution of scores for the two groups are equal.')
                    print('\nMann-Whitney U \'s Statistics (equal variances)=%.3f, p=%.3f' % (stat_u, p_u))
# interpretation
                    if p_u > alpha: 
                        print('Mann-Whitney U \'s test result: There are no differences in medians, samples medians are equal. (fail to reject H0)')
                    else:
                        print('Mann-Whitney U \'s test result: The medians of the two groups are not equal. (reject H0)')  
                        
                else:#Unequal variances
                    stat_u, p_u = mannwhitneyu(*result)
                    print('Levene\'s test result: Unequal variances (reject H0)')
                    print('\nMann-Whitney U \'s Statistics (unequal variances)=%.3f, p=%.3f' % (stat_u, p_u))
# interpretation
                    if p_u > alpha: 
                        print('\nMann-Whitney U \'s Null hypothesis: The distribution of scores for the two groups are equal.')
                        print('Mann-Whitney U \'s test result: The distribution of scores for the two groups are equal. (fail to reject H0)')
                    else:
                        print('\nMann-Whitney U \'s Null hypothesis: The distribution of scores for the two groups are equal.')
                        print('Mann-Whitney U \'s test result: The distribution of scores for the two groups are not equal. (reject H0)')   
                    
  
# ANOVA
            else:    
                stat_l, p_lev = levene(*result)
                print('\nLevene\'s Statistics=%.3f, p=%.3f' % (stat_l, p_lev))
                p_one = np.nan
                if p_lev  > alpha: #Equal variances
                    stat_one, p_one = f_oneway(*result)
                    stat_k, p_h =  kruskal(*result)
                    print('Levene\'s test result: equal variances (reject H0)')
                    print('\nKruskal\'s Statistics(equal variances)=%.3f, p=%.3f' % (stat_k, p_h))
                    print('\nANOVA\'s Statistics(equal variances)=%.3f, p=%.3f' % (stat_one, p_one))
                    if p_one > alpha: 
                        print('\nANOVA\'s Null hypothesis: There are  differences in means')
                        print('ANOVA\'s test result: There are no differences in means, samples are equal. (fail to reject H0)')
                    else:
                        print('\nANOVA\'s Null hypothesis: There are  differences in means')
                        print('ANOVA\'s test result: There are  differences in means (reject H0)')   
                else: # Unequal variances
                    stat_k, p_h  = kruskal(*result)
                    print('Levene\'s test result: Unequal variances (reject H0)')
                    print('\nKruskal\'s Statistics(unequal variances)=%.3f, p=%.3f' % (stat_k, p_h))
                
                if p_h > alpha: 
                    print('\nKruskal\'s Null hypothesis: All sample distributions are equal.')
                    print('Kruskal\'s test result: All sample distributions are equal. (fail to reject H0)')
                else:
                    print('\nKruskal\'s Null hypothesis: All sample distributions are equal.')
                    print('Kruskal\'s test result: One or more sample distributions are not equal. (reject H0)')   

    print('\n\n')

#bi_categorical_cont(df2,'How many books did you read during last 12months?')                    

# =============================================================================
# Readings excerpts:
#https://stats.stackexchange.com/questions/56971/alternative-to-one-way-anova-unequal-variance
#
#@JeremyMiles is right. First, there's a rule of thumb that the ANOVA is robust to heterogeneity of variance so long as the largest variance is not more than 4 times the smallest variance. Furthermore, the general effect of heterogeneity of variance is to make the ANOVA less efficient. That is, you would have lower power. Since you have a significant effect anyway, there is less reason to be concerned here.
#
#Update:
#
#    I demonstrate my point about lower efficiency / power here: Efficiency of beta estimates with heteroscedasticity
#    I have a comprehensive overview of strategies for dealing with problematic heteroscedasticity in one-way ANOVAs here: Alternatives to one-way ANOVA for heteroscedastic data

#The Kruskal-Wallis test is actually testing the null hypothesis that the populations from which the group samples are selected are equal in the sense that none of the group populations is dominant over any of the others or the group population medians are equal             
                    
#That test indicates your data are not normally distributed and the mild skewness indicated by the plots is probably what is being picked up by the test. For typical procedures that might assume normality of the variable itself (the one-sample t-test is one that comes to mind), at what appears to be a fairly large sample size, this mild non-normality will be of almost no consequence at all -- one of the problems with goodness of fit tests is they're more likely to reject just when it doesn't matter (when the sample size is large enough to detect some modest non-normality); similarly they're more likely to fail to reject when it matters most (when the sample size is small).                    

# =============================================================================
# Loading dataset
# =============================================================================

df1 = pd.read_csv('BigML_Dataset_5f50a62795a9306aa200003e.csv') #'D:\\1_Metro College\\Courses\\Python\\Project\BigML_Dataset_5f50a62795a9306aa200003e.csv'

# =============================================================================
# First View of dataset
# =============================================================================
print("===========================Beginning of Analysis======================")

type(df1)

print('\n Dataframe\'s shape: ',np.shape(df1))
    # We see that there are 2832 observations and 14 features

# What are the different data types of df?
print('Features types:\n ',df1.dtypes) # object: string data type
    #We see two integers and 12 objects(strings)

# First and Last Observations
print('Firsts and Lasts Observations')    
print(df1.head(3))
print(df1.tail(3))

# Is there duplicates?
df1.duplicated().sum()

print('\nDuplicated observations:')
print(df1[df1.duplicated()==True] )
    #we see that there is only one observation duplicated, I'll drop it, maintaining the last value.

df2 = df1.drop_duplicates(keep='last')

# How many observations per feature?
df2.count()
    # We can see that some features will have missing values for:
    
#How many Missing values?
df2.isnull().sum()
    # We confirm that there is 390 missing values for:
        # Read any printed books during last 12months? 
        # Read any audiobooks during last 12months? 
        # Read any e-books during last 12months? 
        # Last book you read, you… 
        
# What is the percentage of missing values?
Total = df2.isnull().count().sort_values(ascending=True)        
Missing = df2.isnull().sum().sort_values(ascending=True)   

Percentage_Miss = round((Missing/Total)*100,2)

Summary = pd.concat([Total,Missing, Percentage_Miss],axis=1,keys=['Total','# Missing','% Missing'],sort=True)
          
print('Are there any missing value?')                    
print(Summary)                    
    # We can see that the missing values correspond to approximately 14% of observations                    
    
del Total, Missing, Percentage_Miss, Summary 
# I'll drop the missing ones and confirm if the remaining values are enough to conduct the analysis.
    
# Dropping Missing values
df2 = df2.dropna()

# What is the number of observations after dropping missing values?
df2.count()
    #We see that 2441 observations remain

# =============================================================================
# Segmenting 
# =============================================================================
# Age
min(df2['Age']) # 16
max(df2['Age']) # 93

# Segments:
# less or equal 24 years
# 25 to 44 years
# 45 to 64 years
# 65 or more years 

df2['Age Segmented'] = df2['Age']
df2.loc[df2['Age']> 64,'Age Segmented'] = '65 or more years'
df2.loc[df2['Age']<= 64,'Age Segmented'] = '45 to 64 years'
df2.loc[df2['Age']<= 44,'Age Segmented'] = '25 to 44 years'
df2.loc[df2['Age']<= 24,'Age Segmented'] = '17 to 24 years'
df2.loc[df2['Age']<= 16,'Age Segmented'] = 'less or equal 16 years'

# How many books did you read during last 12months?
# 1 to 25
# 26 to 49
# 50 to 73
# 74 or more

min(df2['How many books did you read during last 12months?']) # 1
max(df2['How many books did you read during last 12months?']) # 97

bins_labels = ['1 to 25','26 to 49','50 to 73','74 or more']
df2['Number of Books (Segmented)'] = pd.cut(df2['How many books did you read during last 12months?'], bins=4,labels=bins_labels)

# =============================================================================
# Transforming String Features to categorical and visualizing them
# =============================================================================
print('\nThese are the categorical features and its categories:\n')
for i in df2.columns:
    if type(df2.loc[0,i]) == str: #checking type of features
        df2[i] = df2[i].astype('category')
        print('These are the categories of \'',i,' \':\n',df2[i].values.categories, '\n')
        
del i
 
# Changing 8 and 9 categories in 'Last book you read, you…' to 'Unknown':
df2['Last book you read, you…'] = df2['Last book you read, you…'].replace(['8','9'],'Unknown')


df2['Last book you read, you…'] = df2['Last book you read, you…'].astype('category')
df2['Last book you read, you…'].values.categories
# =============================================================================
# View of dataset After Segmentation
# =============================================================================
print('Prepared Data(NA\'s and duplicated values have been dropped): ')    
print('\n Dataframe\'s shape: ',np.shape(df2))

print('\nFirsts and Lasts Observations')  
print(df2.head(3))
print(df2.tail(3))

print('\nFirst observation')
print(df2.iloc[1,],'\n')

# =============================================================================
# Univariate Analysis
# =============================================================================

# Numerical Features

# Age and How many books did you read during last 12months?
uni_numerical(df2)

# Categorical Features

# Sex, Race, Marital Status,Education...?

uni_categorical(df2)

# =============================================================================
# Bivariate Analysis
# =============================================================================

# Numerical x Numerical Features
bi_numerical_num(df2,'How many books did you read during last 12months?')   

# Categorical x Numerical Features
bi_categorical_cont(df2,'How many books did you read during last 12months?')

# Categorical x Categorical Features
bi_categorical_cat(df2,'Number of Books (Segmented)')

# =============================================================================
# Others
# =============================================================================
# Consolidating findings for presentation

# Book Media (make a function later)
print('What medias are being used?')
df_media = df2[['Read any printed books during last 12months?',
       'Read any audiobooks during last 12months?',
       'Read any e-books during last 12months?']]

Title = 'Media Types'
mid = df_media.shape[1]//2
#Visual
sns.set_palette("dark") # colorblind   pastel
sns.set(style="white") #whitegrid ticks
sns.set_context("paper", font_scale=1.5)
fig_media, axs = plt.subplots(1,df_media.shape[1], sharey=True)
fig_media.set_size_inches(16, 8)
for i,col in zip(df_media.columns,range(df_media.shape[1])):
    plots = sns.countplot(x=i,data=df_media,ax=axs[col],orient='h')
    for p, label in zip(plots.patches, sorted(df_media[i].unique(),reverse=False)):
                axs[col].annotate(df_media[df_media[i]==label][i].count(), (p.get_x()+0.3, p.get_height()+2))
    if col != 0:
        axs[col].set(ylabel='')
axs[mid].set_title(Title)
plt.show()   

#Verifying the median of Sex
df2.groupby(["Sex"])['How many books did you read during last 12months?'].median()

#loc: only work on index
#iloc: work on position
#at: get scalar values. It's a very fast loc
#iat: Get scalar values. It's a very fast iloc

df2.head()
df2.loc[0:5]['Age']
df2[['Age']][0:5]
df2.iloc[list(range(5)),:]
df2.iat[0,0]
df2.loc[0].at['Age']
df2.loc[0].iat[0]
