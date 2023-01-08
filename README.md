# Alpha Strategy Back Test System
A general backtest system for alpha trading strategy

### Preface
In 2020, I worked as a quantitative research intern in an Alpha strategy team for 9 months. At that time, we developed and backtested our factors in a company-provided system. Since we were a new team, I got a good chance to fully get in touch with many details about the backtest system, including how to deal with suspended and delisted stocks and several factor-to-weight algorithms.  
  
Nowadays, I just finished my master's degree at NYU. The last final project I had was in the **FRE-GY-9713-Active Pricing** course, to build up a daily return prediction model for 100 U.S. equities and develop a trading strategy. Alpha strategy came into my mind first, so I would grab this opportunity to develop my own backtest system, and here is how I structured my codes. Hope these tips can bring you some insights for your own program.

### Structure
Data -> Factor -> Regression -> BackTest

#### 1. Data Storage
There are 2 types of data, stock data and market data. Stock data, like daily return, is various across different companies, while market data, like interest rate, is cross-sectionally the same. So, we need ***DataFrame(index=Date, columns=Ticker)*** to store stock data and ***Series(index=Date)*** to store market data.  
  
In this case, since the number of data I used is limited, I didn't create a new class for data storage. I stored every new data as a column in the member variable, ***self.df(index=[Date, Ticker])***, of ***class "FactorDeveloper"***. Use ***self.df[column].unstack()*** to get the structured stock data.

#### 2. Factor Developing
Since each company would have a different factor score every day, basically speaking, each factor is just a special stock data. You probably need another ***class “Operator”*** to help you achieve various calculation. Use ***.rolling(window)*** to achieve time-series calculation.

#### 3. Daily Regression
The key of an Alpha strategy is how to generate tomorrow's positions with today's factor scores. One easy way is to directly rank today's factor scores, and the weight of each stock is proportional to its rank. This method requires a linear alpha factor, but most of the time, factors are nonlinear. So, we need different machine learning algorithms to achieve nonlinear regression of tomorrow's return (Y) on today's factor scores (X). Use the rank of tomorrow's predicted return to generate positions.  
  
***class "DailyReg"*** takes yesterday's factor scores, today's returns (***df***), and today's factor scores (***td_alpha***) as input. ***df*** is used in training today's regression model, and with ***td_alpha***, we can predict tomorrow's returns. One thing need noticing is the data quality we put into training the model. Some factors have a large number of nan values, so should we drop these factors? Or, how do we fill nan values? Besides, some stocks have no factor value today, so should we fill them with yesterday's values? Or, leave them 0 to cover all of their positions?  
  
Next question is to transform rank into position of each stock. There are 2 factor-to-weight algorithms: Long-Short-Balance (LSB) and Long-Only (Long).
- LSB: 1. unitize: $rank = \frac{rank - min(rank)}{max(rank) - min(rank)}$ -> [0, 1]; 2. LSB: $rank = rank - 0.5$ -> [-0.5, 0.5]; 3. sum to 1: $weight = \frac{rank}{\sum{ABS(rank)}}$
- Long: 1. unitize: $rank = \frac{rank - min(rank)}{max(rank) - min(rank)}$ -> [0, 1]; 2. Long Top p%: $rank = rank$ if $rank > p$ else $0$; 3. sum to 1: $weight = \frac{rank}{\sum{ABS(rank)}}$

#### 4. Daily BackTest
The position of each stock is daily updated by ***class "DailyReg"***, so in ***class "DailyBackTest"***, what we need to do is 1. calculate daily profit, 2. change positions, and 3. calculate trading cost:
- Profit: If all positions remain unchanged, today's total profit $Profit_{t} = \sum{(Position_{t-1}\times (Close_{t}-Close_{t-1}))}$
- Position: For those positions changed, we need to cover the old position, so the price for calculating profit should be $Execution_{t}$ instead of $Close_{t}$. We call this part of difference ***Execution Loss***, and you can design your own execution algorithm in ***function "self.get_exe_price"*** to shrink this loss. Besides, opening a new position with large volume could also bring price impact, and together with the other trading fee, we call them ***Trading Cost***. We assume that price impact $f(Position_{t}) = \sigma_{t} * \sqrt{\frac{|Position_{t}|}{Volume_{t}}}$ for each unit of positions, and you can design your own price impact function in ***function "self.get_trading_cost"***.
- Trading Cost: $Profit = Profit - Execution Loss - Trading Cost$
  
While changing positions, some stocks can have nan values in their new positions or today's close price due to a nan value factor score, suspended or delisted. In this case, a simple method is to remain positions unchanged and assume that there is no profit or loss for these special stocks. Use ***.groupby(Date)[Ticker].count()*** to check the number of tradable stocks every day, and apply your own method to deal with the change of the daily universe. 

### Factor
|Factor|Definition (See *class “FactorDeveloper”*)|
|---|---|
|Beta|The slope of the rolling linear regression between the stock return and the market return with a 1-year window and a 1-quarter half-life.|
|Momentum|The exponential moving average of the stock’s log return with a 1-year window and a 1-month lag.|
|Volatility|The exponential moving standard deviation of the stock return with a 1-year window and a 2-month half-life.|
|Price-Volume Correlation|The rolling covariance of the cross-section rank of the stock’s adjusted price and volume with a 1-month window.|
|Reverse|The rolling max percentage drawdown of the stock’s adjusted price with a 1-month window.|
|Volume Spike|Daily volume divided by its 5-day moving average.|

![](https://github.com/yuba316/Alpha_Strategy_BackTest_System/blob/main/figure/factor_mkt.png)
![](https://github.com/yuba316/Alpha_Strategy_BackTest_System/blob/main/figure/factor_hedge.png)
![](https://github.com/yuba316/Alpha_Strategy_BackTest_System/blob/main/figure/factor_corr.png)
