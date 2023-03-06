# Cash Forecasting for B2B

## Abstract

Unlike B2C where the money and goods are exchanged almost immediately, in B2B, the products and services are provided, and the payment is expected after a certain period of the purchase, and mostly in small remittances. The demand of most of the products and services usually follow a predictive pattern throughout a particular period of time, be it a week, or a year. 

Our aim is to take into consideration all the previous transactions of the company and predict the cashflow for the next two months. 

We intend to use various statistical approaches, along with other Machine Learning and Deep Learning approaches comparing their accuracies. If we further segregate the cashflow by the respective customers of the company, we can even understand the payment patterns of their customers. We first would have to understand the historical data, perform EDA, and identify most promising patterns/columns to work upon. The data would then have to be filtered and preprocessed in order to be accepted by the model and produce desirable output.

Although deep learning networks can be very accurate, they can be very hard to understand, and if a problem occurs, it can be almost impossible to figure out why they happened. Classical ML therefore strikes a balance between accuracy and coherence.

Cash Flow Forecasting would help companies manage their inventories according to the upcoming demand. In B2B , companies can strategize their billing cycles with respect to upcoming payments from their customers. Cash Flow forecasting also helps treasury teams optimize cash flow performance and avoid shortfalls such as customer payments that don’t match existing transactions. In B2C, it can help businesses estimate the demands of the customer, and can also help them identify emerging customer shopping patterns.

## Introduction

Cash Flow Forecasting is the most common way of assessing the flow of money all through a business over a particular time frame. A precise cash flow prediction assists organizations with foreseeing future cash positions, avoid crippling cash shortages, and bring in returns on any money excesses they might have in the most effective way. Cash Flow Forecasting would help companies manage their inventories according to the upcoming demand. In B2B , companies can strategize their billing cycles with respect to upcoming payments from their customers. 

Cash Flow forecasting also helps treasury teams optimize cash flow performance and avoid shortfalls such as customer payments that don’t match existing transactions. In B2C, it can help businesses estimate the demands of the customer, and can also help them identify emerging customer shopping patterns. 

Never before has cash forecasting been more crucial to the financial stability of a company.With an accurate financial projection, your company can foresee future cash shortfalls and minimise missing payments.

Cash flow predictions provide firms the foresight they need to execute corrective actions such as fine-tuning payment and collection procedures, disposing assets, or approaching lenders. Forecasts can assist foresee a surplus as well as lessen the impact of a cash deficit.

Having a large amount of cash sitting around doesn't usually help businesses. Forecasts can assist in spotting future surpluses and help cash managers allocate extra funds effectively. It is essential that firms employ projections and put extra income to use, whether they want to invest or use it to gain a competitive edge. Scenario planning may be used by corporations to foresee the effects of particular investments or actions as they try to allocate spare capital.

Planning your scenarios is a particularly good idea when using automatic cash forecasting. Due to the manual operations needed, organizations sometimes lack the staff or resources to provide customized projections.

Depending on the cash position, a corporation has either cash surplus or cash deficit. For each company, cash forecasting serves a different purpose:

Cash surplus businesses prioritize business growth and M&As and have a lot of cash on hand. Therefore, these businesses may get by with decent accuracy and frequency in their cash flow forecasts. Contrarily, businesses with cash flow problems prioritize prudent cash management, postponing payments, and borrowing at LIBOR rates as opposed to overnight sweeps. To avoid drawing too much from their revolver, they must make precise cash flow predictions and raise the forecasting cadence. However, the accuracy and regularity of cash predictions are constrained when spreadsheets are used.

### 1.1 Problem Definition

Unlike B2C where the money and goods are exchanged almost immediately, in B2B, the products and services are provided, and the payment is expected after a certain period of the purchase, and mostly in small remittances. 

The demand of most of the products and services usually follow a predictive pattern throughout a particular period of time, be it a week, a year, or a decade.

Our aim is to build a ML model for Cash Flow Forecasting to be used in treasury management which can be done either from historical cash receipts or from historical account receivables data and predict the cashflow for the next two weeks.

For the time being, we are using historical bank data to try to find and observe recurring patterns and predict future outcomes using various statistical and deep learning models.

### 1.2 Project Overview/Specifications

Cash Flow Forecasting, depending on the length of the forecasts, can be divided into three categories: Short Term Forecasts: Forecasts that range from 2-5 weeks or about a month into the future; Medium Term Forecasts: Forecasts that range from 5 - 13 weeks, or about 2 months into the future; and finally Long Term Forecasts that often 6months to an year into the future.  

Short Term Forecasts are most useful for short term liquidity planning, while medium term forecasts are useful for interest and debt reduction as well as liquidity risk management. Long Term predictions are generally only useful for the Annual Budgeting Process.

Apart from the length of the predictions, Cash Flow Forecasting can also be divided into 2 types based on the type of forecasting method. They are direct and indirect. The main difference between direct and indirect forecasting is that direct forecasting uses actual cash flow data whereas indirect forecasting relies on projected balance sheets and income statements.

Our project only concerns itself with direct forecasting. For the time being, we’ll only be assessing short term forecasting for small business, but we also plan on using more advanced algorithms for larger organizations, and for a longer forecast period.
Our aim is to build a ML model for Cash Flow Forecasting to be used in treasury management which can be done either from historical cash receipts or from historical account receivables data and predict the cashflow for the next two weeks.

This concept could start from small businesses that may not have detailed information about their customers, demographic data etc, but at least have an invoice management system that keeps track of all the payments made in a day. Here, we’ll be using bank data, which is equivalent.
Businesses deal with customers and have daily transactions. They manage their invoices and create strategies with their current scenario to increase their productivity in their inventories.   

| <img width="660" alt="image" src="https://user-images.githubusercontent.com/43529908/223050684-8fb1b190-3596-4033-abf4-2f9ab149108a.png"> |
| :--------------------------: |
| Figure 1 : Use case diagram |


With the help of those transaction data we fit their business flow and patterns to our model, so we can predict amount flow for upcoming weeks and those records will help them to grow their business.Since we are doing a Time Based Regression, we need to focus on what is the real deal we’re trying to achieve, i.e, Time Based Forecasting.      

Artificial Intelligence is about data. The manner in which we change and feed information into our algorithm — enormously relies upon training results. In time series based Cash Flow Forecasting, the situation is engaged around invoice risk, ML trains to perceive when invoice payment remittance is at risk or how’s the pattern of the business, do they always get some specials payment criteria, is there some seasonality around the dates, are the vendors or customers consistent and are they always on time.

From these assumptions, we can clearly see one thing stands out - Dates Columns. ML algorithm expects numbers as a training feature, it can’t operate with literals or dates. This is when data transformation comes in — out of original data we need to prepare data which can be understood by ML. The question now arises how do we transform the dates into numbers that are interpret-able by our Algorithm? One of the ways is to split the date value into multiple columns with numbers describing the original date (year, quarter, month, week, day of year, day of month, day of week).

Now that we’re done with all basic Date transformations, this might seem everything. But we also need to dig deep to find relationships among Dates and Cash trying to see whether there’s a period around which patterns occur.

### 1.3 Hardware Specification

- Intel core i5-11300H processor : The Intel Core i5-11300H is a mid range SoC for thin and light gaming laptops. It is based on the Tiger Lake H35 generation. It integrates four Willow Cove processor cores and 8 threads. The base clock speed depends on the TDP setting and can vary from 2.6 (28 W TDP) to 3.1 GHz (35 W). The boost of a single and two cores under load can reach up to 4.4 GHz. All four cores can reach up to 4 GHz.                                                                                                                            

- 16GB DDR4 RAM : The larger the RAM the higher the amount of data it can handle, leading to faster processing. DDR4 chips are supposed to help transfer rates between 2133 MT/s (million transfers per second) and 4266 MT/s. By contrast, DDR3 innovation upholds up to 800 to 2133 MT/s. This boost in memory transfer will empower hardware developers to create DDR4 chips with additional strong processors and more proficient gadgets.

- NVIDIA RTX 3050 Ti : This GPU enables the distribution of training processes and can significantly speed machine learning operations.

- 512GB NVMe SSD : Solid State Harddrives enable a faster retrieval of data

- Google Colab
    - CPU: Intel Xeon at 2.30 GHz
    - Available RAM: 12 GB
    - Available Memory : 25 GB
    - GPU : Nvidia K80 12GB

### 1.4 Software Specification

- Numpy v1.21.5 : NumPy is a library for the Python programming language for enormous, multi-dimensional arrays and matrices, alongside a huge assortment of high-level mathematical functions to work on these arrays. The input that we provide in LSTM would be a numpy array. These numpy arrays look like regular lists, but are orders of magnitudes faster than python lists, because Numpy operations are implemented in C, avoiding the general cost of loops in Python, pointer indirection and per-element dynamic type checking. 

- Pandas v1.1.5 : pandas is a python library used for data manipulation and analysis. Specifically, it offers information designs and activities for controlling mathematical tables and time series. In our project, we’ve used pandas for preprocessing, and to engineer relevant columns and features in our dataset. Prophet only takes in a pandas dataframe for predictions.

- Tensorflow v2.8.0 : TensorFlow is a free and open-source library for Artificial Intelligence and Machine Learning. It tends to be utilized across a variety of tasks yet has a particular focus on training, testing and inference of deep neural networks. In our project, we’ll be using tensorflow to implement LSTM.                                                  
                                                                                                                                              
- Prophet v1.1: The Prophet library is planned for making figures for univariate time series datasets. It is easy to use and intended to discover a reasonable arrangement of hyperparameters for the model with an end goal to make capable gauges for information with patterns and occasional design as a matter of course.

- Statsmodels v0.13.2: statsmodels is a Python library that provides functions and classes to implement and assess a wide range of statisticals models, with respect to directing statistical data exploration and statistical tests.

- plotly v5.6.0: Plotly helps us implement interactive charts and maps for Python. This helps to better understand the graphs. You can even pan and zoom in or out in a particular region , and scroll across while zoomed in. There are also options to save the graph as png, and these plots can also be hosted in website where they’ll still be interactive

- Google Colab: Colab notebooks is a Google research project made to assist with spreading AI instruction and exploration. A Jupyter notebook climate requires no arrangement to utilize and runs completely in the cloud.All we need is a browser and execute code on Google's cloud servers, meaning we can leverage the power of Google hardware, including GPUs and TPUs, regardless of the power of your machine.Colab permits us to compose and execute Python in our program, with
    - No configuration 
    - Admittance to GPUs for free
    - Easy sharing

## 2. LITERATURE SURVEY

Many authors have investigated the contemporary rule of treasuries and show how the treasury function is being transformed, considering the current and future scope/challenges faced by corporate treasuries. Every MNC is focused on the transition of a treasurer’s role in context to current business roadblocks and blockers.

The most significant trigger that led to changing of roles of treasurers is the global financial crisis of 2007 to 2011. Since then, the dynamics of treasury has changed. Leading the focus from basic earnings to cash & liquidity. In addition, management of cash has been disrupted, while new risks have conjured up as critical issues.

All these have led to transformation of the treasury world, from performing the basics of cash management and accounting towards a secure strategic liquidity. And this transformation of the treasury world has bought the need for technological changes as a result of shifting business prospects. 

AI could reshape the corporate treasury and bring in the assurance of markets the treasures operate. It brings a lot of credibility regarding to the point where developers have led us to, redefining the world using AI systems.      

Artificial intelligence has the potential to help treasurers deal with regulation changes, increasingly demanding customers and continuing globalization, yet uptake of the technology remains low.

Forecasting is regarded to be among the first administrative tasks. The Bible regularly made reference to prophets and clairvoyants. Today, firms must make more and more estimates since those who don't provide their competitors a clear edge. Today, a lack of foresight is one of the main causes of corporate failure. Because things could be sold solely on a company's reputation in the past, forecasting was not as important. Sentiment is irrelevant in today's more cutthroat marketplace, and companies who don't push themselves to make precise predictions as the cornerstone of their future output will find it increasingly difficult to survive (Lancaster G.A. & Lomas R.A., 1985). Forecasting is important in many aspects of modern business. Organizations must have knowledge of the current situation in order to establish strategies that will take effect in the future (Waters, 2003). This information must be predicted, but despite its significance, progress in many areas has been slow since forecasting is a challenging task (Waters, 2003).

According to the literature, forecasting is defined as follows:

Forecasting is the process of predicting, projecting, or estimating a future occurrence or situation that is outside the control of an organization and acting as the basis for managerial planning. (Golden J. et al. 1994, p.

According to Wikipedia, "forecasting is commonly used to foresee or characterise what will happen" (for example, to sales demand, cash flow, or employment levels), given a certain set of circumstances or assumptions. p.41; Waddell, D., et al., 1994).

Forecasting is the projection of anticipated demand into the future under predetermined environmental circumstances. Moon, M.A., and Mentzer, J.T., 2005, p.

### 2.1 Existing System

Most, if not all, small businesses don’t use Cash Flow Forecasting and only rely on their intuitions, or major events like festivals, pandemic, public curfews and lockdowns. This is good for very rough long term forecasts because these phenomenons are recurring, and moreover, easily observable. For example, businesses near colleges can expect high demand during the months of August and September when the colleges take in new applicants and expect a drop in demand during the months of May and June when colleges usually have summer vacations and most students are away from their homes.

A few major companies do use Cash Flow Forecasting, but these are massive corporations that have millions of gigabytes of data on their customers, and have plenty of resources to spend on the high end servers to implement complicated machine learning models.

The problem with these implementations is their reliance on fast remote servers and extremely complicated models that need all the resources, and make good predictions, but wouldn’t be explainable in case they don’t work.

## 2.2 Proposed System

While this intuition might work for detecting major patterns and give a rough estimate of the upcoming demand. They fail on detecting minor patterns or long term patterns, i.e., patterns that are not immediately obvious to a person. For example, the business owner will be able to tell that the business will boom in August, but they might not be able to tell immediately how much the business will grow compared to their last year's growth and so they wouldn’t be able to tell by exactly how much more they should be keeping the stocks. 

Businesses might also not take into cultural shifts, shopping patterns, and the general shift in people’s mentality. For example, maybe the year on year sales of a particular product is dropping because there is a better alternative available online, or perhaps, the population is collectively leaning towards buying products in bulk so because the combos give a better savings.      

Today's international corporate market, systematic transition from push to pull manufacturing, and emergence of consumer-oriented economies have all contributed to a significantly more complicated world of forecasting (Lapide, 2006). Forecasters are being encouraged to create plans for broadening geographical reach, increasing the number of sales channels, and creating product lines with longer life cycles. Because of this complexity, markets and the corporate environment are more erratic (Lapide, 2006).    

### 2.3 Feasibility Study

The proposed model is lightweight, and specially designed for businesses. Therefore the model also takes into consideration several holidays and other regular transaction patterns. Being lightweight, this could be implemented on existing low spec invoice management systems.

The other, more complex models, that make better predictions, and take more resources can be deployed for the larger businesses that can afford to have the technology. These complex models would be able to take in more parameters, and in the case of B2B, we can segregate frequent customers and make the predictions according to their payment patterns. This will also streamline the remittance process.

Depending on the needs, our feasibility studies may entail further examinations of the market, organizational, technological, and financial elements.
Target markets, customer demographics, the availability of items on the market at the time, the degree of competition, and the location all have an impact on the present and future demand for the suggested goods and services.

Analyzing the organization to determine the management team, their credentials, and the expected staffing needs

Technology analysis: Staffing and training needs, equipment requirements, and business technology requirements.

Start-up costs, continuing costs, revenue projections, finance possibilities, and a profitability analysis are all included in the financial analysis.

Businesses may improve their decision-making by identifying trends and developing future scenarios. The research is typically crucial for allocating funds for the budget, choosing future investments, developing company strategies, making money, and obtaining bank funding.

## 3. SYSTEM ANALYSIS & DESIGN

### 3.1 Data Extraction

The process of converting semi- or unstructured data into structured data is known as data extraction. In other words, this procedure permits the transformation of semi- or unstructured data into structured data. For reporting and analytics, structured data can produce significant insights.

Data extraction is a crucial step in automating the collecting of structured data so that it may be used for additional analysis. The procedure offers the essential information from numerous sources, including contracts, invoices, and communications. These data support process automation and offer insightful analytics for making decisions.

We have 2 datasets, i.e Bank data and ERP data. 

Bank data has the amount in dollar value corresponding to each business day, and ERP data has invoice level data of each transaction and remittance. Those records are utilized by organizations to oversee and coordinate the significant pieces of their organizations.Since we’re starting with small businesses, our focus is on bank data, but for bigger corporations, who have a dedicated treasury team, and have the ability to collect more customer data, we can deploy more advanced predictions.

| No.  | effective_date	| amt         |
|:----:|:---------------|:-----------:|
| 1011 | 2021-12-22	    | 16125490.04 |
| 542  | 2020-02-27	    | 29982419.25 |
| 877  | 2021-06-15     | 15561502.40 |
| 35   | 2018-02-22     | 15315624.31 |
| 48   | 2018-03-13	    | 11292180.10 |

**Table 1: Bank Sample Data**

**effective_date**	 : effective_date is the date in which the particular amount was reflected in the bank account (regardless of when the actual transaction took place).

**amt** : amt represents the total amount that was credit to the bank account corresponding to a particular effective_date.

### 3.2 Design and Test Steps
| <img width="778" alt="image" src="https://user-images.githubusercontent.com/43529908/223055619-4a531bb7-921d-4ddf-8c41-908cb06cd2fd.png"> |
|:--------:|
| Figure 2 : Overview of the machine learning development process |

Machine learning (ML) is a feature that almost all new software products now include. What part does design play in machine learning is a question that sometimes arises for designers. How can designers participate in the process of developing a product powered by machine learning? 

Even though the project's focus was on machine learning for design, the lessons we acquired have a wider range of applications. When working with machine learning, the conventional design tasks—creating a product vision and interacting with stakeholders—apply, but ML also adds new elements to the mix. 

We investigate how design methodologies can be used in machine learning development. Fundamentally, everything revolves around data: gathering it (in large quantities! ), cleaning it up, comprehending it, and ultimately, constructing software on top of it. 

The procedure is as follows: </br>
**Steps**:

1. Data Preprocessing
2. Exploratory Data Analysis
3. Testing Process
4. Base Model Preparation
5. Model Selection  

| <img width="751" alt="image" src="https://user-images.githubusercontent.com/43529908/223056218-047e321c-971e-4247-8f63-126ed21e130a.png"> |
|:----------:|
| Figure 3 : Sequence Diagram |
                                                                                
### 3.3 Data Preprocessing and Feature Engineering

Data preprocessing is a necessary initial step before applying any machine learning apparatus since the algorithms learn from the data and the learning outcome for issue solving is strongly dependent on the right data needed to solve a specific problem, which are referred to as features.

Feature engineering is a machine learning approach that uses data to generate new variables that were not included in the training set. It has the potential to generate new features for both supervised and unsupervised learning, with the objective of simplifying and speeding up data transformations while simultaneously improving model correctness. When dealing with machine learning models, feature engineering is essential. A bad feature will have a direct influence on your model, regardless of the data or architecture.

The data preprocessing and feature engineering that have been performed for the forecast are as following:

- Date-time Conversion
- Manufacturing Additional Helpful Columns
- Removal of extreme amount
- Train-Test-Validation Split

Let’s look at the code and see how each of these steps have been achieved

```
# Converting date into pandas datetime format
bank['effective_date'] = pd.to_datetime(bank['effective_date'])
```
Figure 4 : Date-Time Conversion

By default, the pandas reads the date column as an “object” datatype, i.e., a string in our case. </br>
We need to convert this to pandas-datetime format to efficiently work on all the dates and infer useful features from them.

```
# adding day_number and year to the dataframe

bank['day_number'] = bank['effective_date'].dt.dayofyear
bank['year'] = bank['effective_date'].dt.year
bank['week_number'] = bank['effective_date'].dt.strftime('%U')
bank['month'] = bank['effective_date'].dt.month
```
Figure 5 : Extracting Date into Day, Year, week number and Month 

Next, we extract day_number, year, week_number and month from each date. </br>
**day_number**, in our case, is the day of the year on which the particular date occurs. In pandas, the range for all of these attributes (i.e., day_number, month and week_number) start from 1 and not 0. This implies that January is represented by 1 and not 0. Similarly, the first week is also represented by 1 and not a 0. And the same applies to day_number.

We can make a probability distribution plot (aka histogram plot) to see the spread of the different values of amt.
```
px.histogram(data_frame=bank_new, x='amt', title="Amount Distribution (exluding extreme values)")
```
Figure 6 : Probability Distribution of amount code

|  ![newplot](https://user-images.githubusercontent.com/43529908/223061055-eedcee64-fbd0-464c-8d7e-e1202dd4b652.png) |
|:----------:|
| Figure 7 : Amount Distribution graph |


We can see that a few extreme points are completely throwing off the clarity of the distribution. </br>
To eliminate this, we have to eliminate the extreme points
	
We use InterQuartile Range to trim out the extreme values. We are careful and deliberate in not using the term “outliers”, because these are amounts that have been credited to our bank. An outlier in this case, would mean an unintentional sum of money accidentally credited to a bank account. But here we assuming no such thing happened and all the money is credited for valid reasons.

```
# To trim out extreme amount values

quantile_3 = bank['amt'].quantile(0.75)
print(f'3rd quantile : {quantile_3}')
quantile_1 = bank['amt'].quantile(0.25)
print(f'1st quantile : {quantile_1}')
iqr = quantile_3 - quantile_1
print(f'IQR: {iqr}')
lower_limit = quantile_1 - (1.5*iqr)
print(f'Lower limit: {lower_limit}')
upper_limit = quantile_3 + (1.5*iqr)
print(f'Upper limit: {upper_limit}')


### Trimming out the outliers
bank_new = bank[(bank['amt'] >= lower_limit)&(bank['amt'] <= upper_limit)]
```
Figure 8 : Trimming Extreme amount

But the extreme values shouldn’t be removed without any considerations. Because, in our case, it indicates an abnormally large amount of money being credited to our account. It may be worthwhile for us to look carefully at all the transactions that happened that day, and identify key customers that were responsible for the transactions. 
                                                                                                                                                        
Perhaps they might actually be an outlier, or perhaps they’re a big customer that actually has made a big transaction. In the latter case, we might want to invest more towards those particular high value customers

After removing the extreme points, we now have a better, and a much clearer probability distribution of the amount column

|  ![newplot](https://user-images.githubusercontent.com/43529908/223059166-537fd77d-a37a-469c-8dbf-efd4347d98a8.png) |
|:----------:|
| Figure 9 : Trimming Extreme Amount graph |

We can see that the distribution is right-skewed (or positively skewed). 

This means that the **mean** of the distribution would overestimate the most common values, so a **median** would be the most appropriate measure to understand the “average” of the data.

In terms of our business, this indicates that we can expect frequent small profits (around $15 Million - $20 Million a day) and fewer big profits.

```
# Train Test Validation Split
# We will take a 85:10:5 split of Train:validation:Test 

bank_train = bank_new[:int(bank_new.shape[0]*0.85)]
bank_val = bank_new[int(bank_new.shape[0]*0.85): int(bank_new.shape[0]*0.95)]
bank_test = bank_new[int(bank_new.shape[0]*0.95):]
```
Figure 10 : Splitting into Train, Test and Validation

We split the data into train, test and validation. We shall train the data on the training set, test the data on the validation set. And using the model with the least loss in the validation set, we will calculate the loss on the test set to finally report the loss results of our final model on completely unseen data.                                                                                                                                            


### 3.4 Exploratory Data Analysis

Under Exploratory Data Analysis (EDA), we will do as the name suggests, i.e., explore the data. We will try and identify any perceivable patterns, trends, or seasonalities.

Exploring and understanding the data is perhaps the most important step in any Machine Learning project. There is a popular saying in the Machine Learning field that says, “Garbage in, Garbage out”. Which means, if we simply put raw data without any processing, or understanding, the results generated through the models would be equally random and unusable.

Thus, by simply creating a few basic graphs, we may gain a thorough understanding of our dataset. It just depends on the queries you want to pose to the data. You'll find all the solutions in the plotting.

To Start, let’s look at the time series as it is, and try and find any patterns

```
fig = px.line(bank_train, x="effective_date", y="amt", title='bank Trend')
fig.show()
```
Figure 11 : Bank amount trend code

| ![newplot](https://user-images.githubusercontent.com/43529908/223062934-6ad69850-c7f4-46ad-9144-b4fdbdf33179.png) |
|:----------:|
| Figure 12 : Bank amount trend graph |

This all looks like randomly generated noise. We aren’t able to identify any trends or seasonality. The overall flow might look somewhat like a sine wave, but even that isn’t clear enough to be investigated further.

Now let’s check the year-on-year graphs on a daily basis.

Year-on-year graphs, help us compare the patterns of any particular year to the patterns of any other particular year.
```
# YoY Graph on a daily basis

bank_yoy = pd.pivot_table(bank_new, values='amt', index='day_number', columns='year', aggfunc='sum')
fig = bank_yoy.plot(title='Bank YoY Graph on a daily basis', labels=dict(value="log(money)"))
fig.update_yaxes(tickprefix="$")
```
Figure 13 :  Bank Year-On-Year graph on a daily basis code


| ![newplot](https://user-images.githubusercontent.com/43529908/223077379-e3bcc3ba-0ac8-4300-99bc-052be2c84439.png) |
|:----------:|
| Figure 14 : Bank Year-On-Year graph on a daily basis |


In an ideal case, we would be able to predict on a daily basis, and keep the exact stocks that are needed in the near future; Or in case of B2B (as is our case here), we would be able to predict when to get the remittances as well. But, we can never know exactly how much demand we can expect in a day. Even in this bank data, we can clearly see the  noise in the data. There are absolutely no visible seasonalities or trends. This is a good example of cases where applying a Machine Learning model would be a complete waste of time.

Now let’s look at year-on-year graph on a weekly basis

```
# YoY Graph on a weekly basis

bank_yoy = pd.pivot_table(bank_new, values='amt', index='week_number', columns='year', aggfunc='sum')
fig = bank_yoy.plot(title='Bank YoY Graph on a weekly basis', labels=dict(value="log(money)"))
fig.update_yaxes(tickprefix="$")
```
Figure 15 : Bank Year-On-Year graph on a weekly basis code

| ![newplot](https://user-images.githubusercontent.com/43529908/223066423-aadc59a3-9ff7-4b0f-b7f3-2c3ec822ee87.png) |
|:----------:|
| Figure 16 : Bank Year-On-Year graph on a weekly basis |


For the year-on-year graph on the weekly basis, we see quite a lot of repeating patterns. For example we can see a consistent rise year on year on week 39, and a consistent dip in week 21.

We can see that there are some observable patterns, and this looks to be promising input for forecasting. So let’s investigate further into it.
Time series is consists of  3 components :

1. trend : continuous increases or decreases in a metric's value.
2. seasonality : the recurring patterns in a particular time period
3. residual : whatever is left after removing trend and seasonality from the time series

```
weekly_trend = np.array(bank_train.groupby(['year', 'week_number'])['amt'].sum())
ts_dicomposition = seasonal_decompose(x=weekly_trend, model='multiplicative', period=53)
trend_estimate = ts_dicomposition.trend
seasonal_estimate = ts_dicomposition.seasonal
residual_estimate = ts_dicomposition.resid
# Plotting the time series and it's components together
fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_figheight(10)
fig.set_figwidth(20)
# First plot to the Original time series
axes[0].plot(weekly_trend, label='Original') 
axes[0].legend(loc='upper left');
# second plot to be for trend
axes[1].plot(trend_estimate, label='Trend')
axes[1].legend(loc='upper left');
# third plot to be Seasonality component
axes[2].plot(seasonal_estimate, label='Seasonality')
axes[2].legend(loc='upper left');
# last last plot to be Residual component
axes[3].plot(residual_estimate, label='Residuals')
axes[3].legend(loc='upper left');
```
Figure 17 : Time series decomposition on weekly basis code


| ![Unknown](https://user-images.githubusercontent.com/43529908/223069058-c4788c47-0cf8-486c-8809-5a499d813862.png) |
|:----------:|
| Figure 18 : Time series decomposition on weekly basis graph |                                                           

After doing a time-series decomposition, we are able to make a certain trend on a weekly basis, how it’s first increasing, and then decreasing. This is probably due to the reduced demand of goods, and reduced business due to the pandemic and the lockdown.
There’s a very rough seasonality that we can also see. And the fact that we cannot see any patterns in the residual points us to the fact that our time-series decomposition is done correctly.
Let’s also have a look at the year-on-year graph on a monthly basis
```
# YoY Graph on a monthly basis

bank_yoy = pd.pivot_table(bank_new, values='amt', index='month', columns='year', aggfunc='sum')
fig = bank_yoy.plot(title='Bank YoY Graph on a mothly basis', labels=dict(value="log(money)"))
fig.update_yaxes(tickprefix="$")
```
Figure 19 : Bank Year-On-Year graph on a monthly basis code

|  ![newplot](https://user-images.githubusercontent.com/43529908/223075786-a3683aec-5214-44d6-a840-58eeb1565768.png) |
|:----------:|
| Figure 20 : Bank Year-On-Year graph on a monthly basis graph  |      
                                                                                                                                                
While the graph looks really clean, and appears to be a good input for machine learning models, we have to consider the practicality of the models we’d be using. While this type of monthly predictions would be fairly accurate, the practicality of such a prediction is questionable.

### 3.5 Testing Process

Since we are trying to estimate an amount of money, this time-series forecasting comes out to be a regression problem. This means that we will not be able to measure the accuracy of the predictions. Our best approach would be to calculate the loss. This would indicate how far away from our predicted data is from the actual data. 

Testing may improve ML by verifying that the system operates as intended when code from the library is included, hence enhancing model quality. Test cases are used to assess whether or not the system under test meets the requirements and functions properly.

There are two main methods available for loss calculation

1. Mean Absolute Error (L1 Loss)
2. Root Mean Squared Error (L2 Loss)


**Mean Absolute Error** is the average absolute error between actual and predicted values.The error between two observations reflecting the same phenomena is measured by this metric. Comparisons of expected against observed data, subsequent time against starting time, and one measuring technique against an alternate measurement technique are a few examples of Y vs X. The MAE is determined by dividing the total absolute errors by the sample size.
We should be aware that other formulations could use relative frequencies as weight factors. Similar to the scale used to measure the data, the mean absolute error utilizes the same scale. Since this is a scale-dependent accuracy metric, series employing various scales cannot be compared using this measure.  

**Root Mean Squared Error** is the square root of the mean squared error between the predicted and actual values.

For our purpose, we’ll be using Root Mean Squared Error (referred to as RMSE from here on). Let’s elaborate on RMSE now. 

RMSE is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far away from the regression line data points are; RMSE is a proportion of how spread out these residuals are. As such, it lets you know how focused the information is around the line of best fit.

It’s because in our scenario, A difference of $20 Million isn’t just twice as bad as a difference of 10 Million. A difference of $20 Million is much more severe than a difference of $10 Million. We would like to give more weights to observations that are further from the mean and RMSE is much more sensitive to observations that are further from the mean. 

### 3.6 Base Model Implementation

Baseline models are simple by definition. They typically have a small number of trainable parameters and may be easily and quickly fitted to your data.
Thus, simple models are the following in engineering:
faster training, providing immediate performance feedback.
Better studying means that the most of the faults you run into will either be simple model bugs or will reveal a problem with your data.
They are quicker for inference, thus deploying them doesn't call for a lot of infrastructure engineering and won't add to latency.
The optimum time to decide what to do next is after you've constructed and implemented a baseline model.
A baseline provides context for a more sophisticated model.
The performance that is trivially achievable is a benchmark that you would expect any model could surpass. The accuracy you would obtain while inferring the most prevalent class in a classification task could serve as an illustration of this value.
Human performance, or how well a person can complete this task, is the level. When compared to humans, computers are significantly better at some skills (like playing Go) than others (like writing poetry). Knowing how skilled a human is at something might help you calibrate your algorithm's expectations in advance, but as the human/computer gap varies greatly by field, some literature research may be necessary.
Before choosing the best model , We made the base model which is the simplest model. The complex models we used after this are supposed to be much better than this. So based on the base model, we decide whether the complex model should be implemented or not. and we can compare the time taken to predict, accuracy and the RMSE value .

Figure 21 : Median Growth Rate Code
Baseline Model with Median Growth Rate
In layman's words, the growth rate of a function f(x) refers to how quickly the value of f(x) rises or decreases as the value of x increases. For example, if f(x)=x, the function rises by one unit for every unit increase in x, but if f(x)=10x, the function increases by ten units for every unit increase in x.
When an algorithm's input increases, the number of steps it takes can be estimated using growth functions.
In our case, the growth rate would be the increase in year x with respect to the previous year x-1. So the growth rate of the year 2021 would be the amount earned in year 2021 divided by the amount earned in the year 2020. Similarly, we can calculate the growth rate of a particular week of an year, by contrasting it against that particular week of the previous year.
From here on, when we say growth rate of any year, we mean growth rate of that year on a week by week basis
In the Baseline model, we first calculated the base growth rate for the year 2019, 2020 and 2021. Then we calculated the median growth rate by taking the median of all the previous growth rates.
In a previous section, we had mentioned earlier why median is a more appropriate measure compared to a mean in our case. 
After that by multiplying the median growth rate with the amount in 2021, we can predict the amount for 2022.

 Figure 22 : Median Growth Rate predictions (2022_pred)

We can already see a big difference in week 7. Investigating it further, we found that only 1 day of week 7 is captured in 2022.So we are eliminating this entire week.


Figure 23 : Week 7 Investigation (Median Growth Rate) 

Figure 24 : Loss Calculation (Median Growth Rate) code
                                                                                                                                                                                     And this way, our final loss in the median-growth rate model comes out to be around $26 Million
3.7 Model Selection

The process of selecting the model that best generalizes is referred to as model selection. To mimic unknown data, training and validation sets are utilized. Overfitting occurs when our model performs well on our training dataset but performs badly in general. When our model performs badly on both our training dataset and unknown data, we have underfitting. We may use cross-validation techniques to determine if our model generalizes successfully. RMSE, or root mean squared error, is a frequent assessment statistic.
We are gonna use forecasting models of different categories i.e Statistical models and deep learning models. So based on the loss calculations we will choose the model to predict. 
Here some forecasting techniques are :
SARIMA
SARIMA, which stands for Seasonal-AutoRegressive Moving Average model, includes the forecast's seasonality component. The significance of seasonality is pretty obvious, and ARIMA fails to implicitly capture that information.
SARIMA is an enhancement to ARIMA that allows for direct modeling of the seasonal component of the series. It introduces three new hyperparameters to describe the autoregression (AR), differencing (I), and moving average (MA) for the seasonal component of the series, as well as a new parameter for the seasonality period. There are three trend components that must be configured while establishing a SARIMA.


They are identical to the ARIMA model, specifically:
Trend autoregression order (p)
Trend difference order (d)
Trend moving average order (q)

But in addition of all the trend parameters, SARIMA also needs to be configured with some seasonal parameters that are as follows:
Seasonal autoregressive order (P)
Seasonal difference order (D)
Seasonal moving average order (Q)
The number of time steps for a single seasonal period. (m)

 Prophet
Prophet is a system for predicting time series data based on an additive model in which non-straight patterns are fit with annual, weekly, and daily irregularity, as well as occasional effects. It works well with time series that have strong occasional impacts and a few instances of verified data.
On a weekly or annual basis, it is utilized to create goals and distribute resources. Prophet is totally automated so that novices in data forecasting can quickly provide accurate predictions, but it also permits manual tweaking so that professionals may enhance outcomes by incorporating specific expertise. Prophet may automatically discover changes in trends by choosing change points from data and modeling yearly components using the fourier series, which works well with time series that contain seasonal impacts and numerous seasons of historical data. Prophet, however, is resilient to missing data, changes in trend, and generally handles outliers well. Prophet was open sourced by Facebook's core data science team in early 2017. Having been fully optimized for the business duties they confronted at Facebook.

LSTM
Long Short-Term Memory (LSTM) is an artificial neural network used in the fields of artificial intelligence and deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections.
LSTM networks were created specifically to address the issue of long-term dependency. This property allows LSTMs to process entire data sequences (e.g., time series) without having to treat each point in the sequence independently, but rather by retaining useful information about previous data points in sequence to aid in the processing of new data points. As a result, LSTMs excel at processing data sequences such as text, speech, and general time series. 
This pattern, which exists every 12 periods of time, can be learned by an LSTM network. It does not simply use the previous prediction, but rather keeps a longer-term context in mind, which helps it overcome the long-term dependency issue that other models face. It is worth noting that this is a very simplified example, but LSTMs become increasingly useful when the pattern is separated by much longer periods of time (as in long passages of text, for example).

3.7.1. SARIMA
Before we implement SARIMA, we need to understand stationary time series.
A stationary time series is a time series, whose values don’t change over time, i.e., the time series that has no general trend. In these cases summary statistics, like mean and variance of the observation remains consistent over time
In order to eliminate additive seasonal effects, a seasonal ARIMA model employs differencing at a lag equal to the number of seasons (s). The lag s differencing introduces a moving average term, just like lag 1 differencing does to eliminate a trend. Autoregressive and moving average terms are present in the seasonal ARIMA model at lag s.
Statistical models like ARIMA and SARIMA require time series to be stationary to be effective.
We can check whether a time series is stationary by using statistical tests like Augmented Dickey-Fuller test. The null hypothesis is that the time series is not stationary and its alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary. 
We interpret the result using p-value. A p-value below 0.05 (or less than 5%) suggests that our null-hypothesis is wrong (i.e., time-series is stationary) and if it is more than 0.05 then it means the null hypothesis holds (i.e., the time-series is not stationary)


Calculating Augmented Dicky-Fuller test on our dataset
Figure 25 : Calculation of Augmented Dicky-Fuller test

We get the test statistic value of -6.663. Usually the more negative the test statistic, the more likely we are to reject the null hypothesis. 
Since the p-value is also less than 0.05, this means that our null-hypothesis is wrong, and our time-series is stationary.
We now implement SARIMA, and do a grid search for the best hyper-parameters to get the lowest loss.
We start by defining a single step of the model, that is, a function that takes in a single configurations and fits the model to the data

  Figure 26 : SARIMA forecast code

We need to find the RMSE value for each combination of parameters, so we define our loss function as well


Figure 27 : RMSE code
We would also need to split our dataset into train and test set, so we made a train_test_split() function to achieve that 

Figure 28 : Splitting univariant dataset

First, the train test split() function divides the supplied univariate time series dataset into training and test sets. Then the number of observations in the test set are enumerated. We make a one step forecast and for all the history we fit a model for each. The history and process is repeated for the true observation for the time step.to make a prediction we fit a model by calling the sarima_forecast() function. Finally the measure_rmse() function is called and calculates  the error score by comparing all one-step forecasts to the actual test.





Figure 29 : Validation for univariate data code

We can then call the walk_forward_validation() function with all the combinations of parameters.
Some of these combinations might throw an error, or a warning, so we’ve also suppressed that. So we’ve encapsulated the walk_forward_validation() function within a try-except block. The score_model() function below does exactly this. it contains the walk_forward_validation() function within the try-except block and returns the model configurations and the loss associated with that particular combination.

Figure 30 : Parallel execution of grid search code
After this, we’ll have to loop through all the different combinations of parameters and call the score_model() function over and over again, with all those different combinations of parameters. This is arguably the most crucial step of the entire grid-search process.
Figure 31 : Grid Search configuration code
We will be using the joblib library to evaluate model configurations in parallel, so as to streamline this process. A list of modelconfiguration(list of lists), and number of time steps to use in time set, the grid_search() function below implements this behavior given an univariate time series dataset.
Now finally, all we’re left to do is to make all the model configurations we want to test, and asrima_configs() function is meant to do exactly that
Figure 32 : SARIMA configuration code

 
Figure 33 : Main function calling functions and printing results
Finally we call each function and The lowest RMSE we found across all the model configurations was $9,268,643.93

3.7.2. Prophet
Prophet is an open-source time series forecasting library developed by Facebook (now Meta) for ease of use by any person regardless of their technical knowledge in the Machine Learning domain.
On a weekly or annual basis, it is utilized to create goals and distribute resources. Prophet is totally automated so that novices in data forecasting can quickly provide accurate predictions, but it also permits manual tweaking so that professionals may enhance outcomes by incorporating specific expertise. Prophet may automatically discover changes in trends by choosing change points from data and modeling yearly components using the fourier series, which works well with time series that contain seasonal impacts and numerous seasons of historical data. Prophet, however, is resilient to missing data, changes in trend, and generally handles outliers well. Prophet was open sourced by Facebook's core data science team in early 2017. Having been fully optimized for the business duties they confronted at Facebook.
Prophet builds the model by finding out the best fitting line which is usually described by a combination of an overall growth trend, yearly seasonality, weekly seasonality and holiday effects.
The input to Prophet is a dataframe with two columns: ds and y. The ds (datestamp) column ought to be of a format expected by Pandas; YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y section should be numeric, and should represent the measurement we wish to foresee.
We initialize a of the Prophet class and afterward call its fit and forecast functions.
Figure 34 : Prophet code
You can get a reasonable dataframe that stretches out into the future a predetermined number of days utilizing Prophet.make_future_dataframe helper method

Figure 35 : Prophet predictions (yhat)
We can plot the forecasts to get a more visual understanding of how good the predictions are.

Figure 36 : Plot forecast(graph)
We then compare the predictions with the validation data and obtain the RMSE value of $2,050,424 

Figure 37 :  Loss Calculation (Prophet )
The amount loss in the Prophet model is very much smaller than the loss amount in the median growth rate. because it is optimized specially for the business.

3.7.3. LSTM
LSTMs are a popular deep learning model, used for any sequential data, be it unsegmented, connected handwriting recognition, guessing the next word, or speech recognition. LSTM addresses 1 major problem that typical RNNs face. That is, dealing with long term dependencies.
Let’s look at how it’s being implemented
We essentially have a series of amounts, and using that we have to predict what would come next (or at least some number close to the actual number). The input for the LSTM therefore looks something like this
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
To apply this series to LSTM, we need to sample small series from the total series, and repeatedly predict the next number. This would look something like this

Input
Output
[1, 2, 3, 4, 5]
6
[2, 3, 4, 5, 6]
7
[3, 4, 5, 6, 7]
8
…
…
[10, 11, 12, 13, 14]
15


Table 2 : Sample LSTM training data
So, we write a function for doing this

Figure 38 : function to sample the data for LSTM
This takes in the series of numbers, and n_steps. n_steps is the length of the input list. It then samples all the data, and appends it to a list of lists that have all the input lists, and output numbers. 
We then apply that function to our training and validation data

Figure 39 :  Sampling of training and validation data
We then define the model
Figure 40 : LSTM model code
The network has a hidden layer with 50 LSTM blocks or neurons, and an output layer that makes a single value prediction. The default sigmoid activation function is used for the LSTM blocks. The network is trained for 60 epochs (20 epochs running 3 times). The final training looked like this : 

Figure 41 : LSTM model training
LSTM is rather suited for much larger multivariate data, and moreover, we don’t have the hardware to support longer lookbacks for LSTM, therefore, to set a baseline, we chose 50 units, and only 20 epochs for training.
Our final loss comes out to be 120525556809728 which is the mean squared error.
Taking square root of the loss, we get an RMSE of $10,978,413.21,  which is a lot, but it’s still better than our baseline model with the median growth rate. There is potential in the deep learning models, but they can only be used by massive corporations which have a lot of data and the processing power to run the complex deep learning models.

