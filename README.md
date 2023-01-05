# Alpha Strategy Back Test System
A general backtest system for alpha trading strategy

### Preface
In 2020, I worked as a quantitative research intern in an Alpha strategy team for 9 months. At that time, we developed and backtested our factors in a company-provided system. Since we were a new team, I got a good chance to fully get in touch with many details about the backtest system, including how to deal with suspended and delisted stocks and several factor-to-weight algorithms.  
  
Nowadays, I just finished my master's degree at NYU. The last final project I had was in the Active Pricing course, to build up a daily return prediction model for 100 U.S. equities and develop a trading strategy. Alpha strategy came into my mind first, so I would grab this opportunity to develop my own backtest system, and here is how I structured my codes. Hope these tips can bring you some insights for your own program.

### Structure

### Factor
|Factor|Definition (See class “FactorDeveloper”)|
|---|---|
|Beta|The slope of the rolling linear regression between the stock return and the market return with a 1-year window and a 1-quarter half-life.|
|Momentum|The exponential moving average of the stock’s log return with a 1-year window and a 1-month lag.|
|Volatility|The exponential moving standard deviation of the stock return with a 1-year window and a 2-month half-life.|
|Price-Volume Correlation|The rolling covariance of the cross-section rank of the stock’s adjusted price and volume with a 1-month window.|
|Reverse|The rolling max percentage drawdown of the stock’s adjusted price with a 1-month window.|
|Volume Spike|Daily volume divided by its 5-day moving average.|

![](https://github.com/yuba316/Alpha_Strategy_BackTest_System/blob/main/figure/factor_mkt.png)
