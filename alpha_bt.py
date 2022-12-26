# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

path = r"E:\NYU\7913-Active Portfolio\project\data"


#%% 1. read in and process data

df = pd.read_table(os.path.join(path, "prices.txt"), sep="\t")

def split_row(string):
    return [i for i in string.split(" ") if i != ""]

columns = split_row(df.columns[0])
df["row"] = df.iloc[:,0].apply(split_row)
df[columns] = df["row"].apply(pd.Series)
df = df.iloc[:,2:]
df = df.applymap(lambda x: np.nan if x=="." else x)
df.drop([32765, 65531, 98297], inplace=True)
for i in columns[3:]:  # price, volume should all be numeric
    df[i] = pd.to_numeric(df[i])
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df.set_index(["date", "PERMNO"], inplace=True)
df["STD"] = df.groupby("PERMNO")["PRC"].pct_change().rolling(21).std()


#%% 2. multi-factor model

class Operator:
    def __init__(self):
        self.author = "Joey Zheng"
    
    @staticmethod
    def weight(window, half):
        return np.power(0.5, (np.arange(window) + 1) / half)
    
    @staticmethod
    def wls(Y, X, window, half):
        X = sm.add_constant(X)
        w = Operator().weight(window, half)
        try:
            model = regression.linear_model.WLS(Y, X, w, missing="drop")
        except:
            return np.nan
        else:
            res = model.fit()
            return res.params[0]
    
    @staticmethod
    def rolling_wls(Y, X, window, half):
        Y = Y.loc[X.index].values
        X = X.values
        n = len(Y)
        res = np.repeat(np.nan, n)
        for i in range(window, n):
            res[i] = Operator().wls(Y[i-window:i], X[i-window:i], window, half)
        return res
    
    @staticmethod
    def ols(Y, X):
        X = sm.add_constant(X)
        try:
            model = regression.linear_model.OLS(Y, X, missing="drop")
        except:
            return np.repeat(np.nan, X.shape[1])
        else:
            res = model.fit()
            return res.params
    
    @staticmethod
    def mdd(X):
        return max((np.maximum.accumulate(X) - X) / X)


class FactorDeveloper:
    def __init__(self, df):
        self.df = df.copy()  # all universe data
        self.mkt_df = self.df.groupby("date").count()[["TICKER"]]  # mkt data
    
    def data_adj_price(self):
        self.df["ADJPRC"] = self.df["PRC"] / self.df["CFACPR"]
        return
    
    def data_rtn(self):
        if "ADJPRC" not in self.df.columns:
            self.data_adj_price()
        self.df["RTN"] = self.df.groupby("PERMNO")["ADJPRC"].pct_change()
        return
    
    def data_mkt_rtn(self):
        if "RTN" not in self.df.columns:
            self.data_rtn()
        self.mkt_df["MKTRTN"] = self.df.groupby("date")["RTN"].mean()
        return
    
    def risk_beta(self, window=252, half=63):
        if "MKTRTN" not in self.df.columns:
            self.data_mkt_rtn()
        Y = self.df["RTN"].unstack().copy()
        X = self.mkt_df["MKTRTN"].copy()
        res = Y.apply(lambda x: Operator().rolling_wls(x, X, window, half))
        res = res.apply(lambda x: (x-x.mean())/x.std(), axis=1)
        return -res
    
    def risk_mom(self, window=252, lag=21):
        if "RTN" not in self.df.columns:
            self.data_rtn()
        res = self.df["RTN"].unstack().copy()
        res = np.log(res + 1)
        alpha = 2 / (1 + window)
        res = res.ewm(min_periods=window, adjust=False, alpha=alpha).mean()
        res = res.shift(lag)
        res = res.apply(lambda x: (x-x.mean())/x.std(), axis=1)
        return -res
    
    def risk_vol(self, window=252, half=42):
        if "RTN" not in self.df.columns:
            self.data_rtn()
        res = self.df["RTN"].unstack().copy()
        res = res.ewm(min_periods=window, adjust=False, halflife=half).std()
        res = res.apply(lambda x: (x-x.mean())/x.std(), axis=1)
        return res
    
    def alpha_pvcov(self, window=21):
        if "ADJPRC" not in self.df.columns:
            self.data_adj_price()
        res = self.df[["ADJPRC", "VOL"]].copy()
        res = res.groupby("date").rank()
        res["PVCOV"] = res.reset_index().groupby(
            "PERMNO")[["ADJPRC", "VOL"]].rolling(window).cov().unstack()[
                "ADJPRC"]["VOL"].values
        res = res["PVCOV"].unstack()
        res = res.apply(lambda x: (x-x.mean())/x.std(), axis=1)
        return -res
    
    def alpha_reverse(self, window=21):
        if "ADJPRC" not in self.df.columns:
            self.data_adj_price()
        res = self.df["ADJPRC"].unstack().rolling(
            window).apply(lambda x: Operator().mdd(x))
        res = res.apply(lambda x: (x-x.mean())/x.std(), axis=1)
        return res
    
    def alpha_volspike(self, window=5):
        res = self.df["VOL"].unstack().copy()
        res = res / res.rolling(window).mean()
        res = res.apply(lambda x: (x-x.mean())/x.std(), axis=1)
        return res


factor = FactorDeveloper(df)
beta, mom, vol = factor.risk_beta(), factor.risk_mom(), factor.risk_vol()
pvcov, reverse, volspike = factor.alpha_pvcov(), factor.alpha_reverse(), factor.alpha_volspike()


#%% 3. regression

class DailyReg:
    def __init__(self):
        self.author = "Joey Zheng"
        self.last_df = None
        self.last_alpha = None
    
    def update(self, df, td_alpha):
        self.X = df.iloc[:,1:].copy()
        X_columns = self.X.columns[len(self.X) - self.X.isna().sum() >= 30]
        alpha_columns = td_alpha.columns[len(td_alpha) -
                                         td_alpha.isna().sum() >= 30]
        columns = list(set(X_columns).intersection(set(alpha_columns)))
        self.X = self.X[columns].dropna()
        self.td_alpha = td_alpha[columns]
        self.td_alpha = self.td_alpha.fillna(self.td_alpha.mean()).fillna(0)
        self.Y = df.loc[self.X.index,"rtn"].copy()
        self.Y = self.Y.fillna(self.Y.mean()).fillna(0)
    
    def get_reg(self, model="Combo", merge=None):
        if model == "Combo":
            return self.Combo()
        if model == "OLS":
            return self.OLS()
        if model == "Ridge":
            return self.Ridge()
        if model == "RF":
            return self.RandomForest()
        if model == "GBDT":
            return self.GBDT()
        if model == "Merge":
            return self.Merge(merge)
        if model == "RankMerge":
            return self.RankMerge(merge)
    
    @staticmethod
    def LSB(alpha, pct=0):
        rank = alpha.rank()
        mini, maxi = rank.min(), rank.max()
        rank = (rank - mini) / (maxi - mini) - 0.5
        return rank.apply(lambda x: x if abs(x)>=0.5*pct else 0)
    
    @staticmethod
    def LONG(alpha, pct=0.5):
        rank = alpha.rank()
        mini, maxi = rank.min(), rank.max()
        rank = (rank - mini) / (maxi - mini)
        return rank.apply(lambda x: x if x>=0.5 else 0)
    
    @staticmethod
    def orthogonalize(temp):
        X = temp.fillna(0).values
        D, U = np.linalg.eig(np.dot(X.T, X))
        S = np.dot(U, np.diag(D**(-0.5)))
        return pd.DataFrame(X @ S @ U.T, columns=temp.columns,
                            index=temp.index)
    
    def Combo(self):
        return self.td_alpha.rank().mean(axis=1)

    def OLS(self):
        reg = LinearRegression()
        reg.fit(self.X, self.Y)
        res = reg.predict(self.td_alpha)
        return pd.Series(res, index=self.td_alpha.index)
    
    def Ridge(self):
        reg = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1])
        reg.fit(self.X, self.Y)
        res = reg.predict(self.td_alpha)
        return pd.Series(res, index=self.td_alpha.index)
    
    def RandomForest(self):
        reg = RandomForestRegressor(n_estimators=100, max_depth=None,
                                    max_leaf_nodes=None)
        reg.fit(self.X, self.Y)
        res = reg.predict(self.td_alpha)
        return pd.Series(res, index=self.td_alpha.index)
    
    def GBDT(self):
        reg = GradientBoostingRegressor(learning_rate=0.01, n_estimators=100,
                                        max_depth=None, tol=1e-8)
        reg.fit(self.X, self.Y)
        res = reg.predict(self.td_alpha)
        return pd.Series(res, index=self.td_alpha.index)
    
    def Merge(self, model=None):
        model = ["OLS", "Ridge", "RF", "GBDT"] if model is None else model
        model_list = []
        for i in model:
            model_list.append(self.get_reg(i))
        res = pd.concat(model_list, axis=1)
        return res.mean(axis=1)
    
    def RankMerge(self, model=None):
        model = ["OLS", "Ridge", "RF", "GBDT"] if model is None else model
        model_list = []
        for i in model:
            model_list.append(self.get_reg(i))
        res = pd.concat(model_list, axis=1)
        return res.rank().mean(axis=1)


class RegTest:
    def __init__(self, df, factor, model, start=None, end=None):
        self.df = df
        self.factor = factor
        self.model = model
        self.univ = (self.df["PRC"] / self.df["CFACPR"]).unstack().loc[start:
                                                                       end]
        self.rtn = self.univ.pct_change()
        self.date = self.univ.index.values
    
    def get_X(self, idx):
        X = pd.concat([self.rtn.loc[self.date[idx]]] +
                      [self.factor[i].loc[self.date[idx-2]] for i in
                       self.factor], axis=1)
        X.columns = ["rtn"] + list(self.factor.keys())
        return X
    
    def get_td_alpha(self, idx):
        td_alpha =  pd.concat([self.factor[i].loc[self.date[idx]] for i in
                               self.factor], axis=1)
        td_alpha.columns = list(self.factor.keys())
        return td_alpha
    
    @staticmethod
    def get_score(y_pred, y_true):
        score = {}
        df = pd.concat([y_pred, y_true], axis=1)
        score["corr"] = df.corr().iloc[0,1]
        score["MSE"] = np.square(df.diff(axis=1).iloc[:,1]).mean()
        return score
    
    def execution(self):
        td_reg = DailyReg()
        score = []
        n = len(self.date)
        for i in range(2, n-2):
            X = self.get_X(i)
            td_alpha = self.get_td_alpha(i)
            td_reg.update(X, td_alpha)
            rtn_pred = td_reg.get_reg(self.model)
            rtn_true = self.rtn.loc[self.date[i+2]]
            score.append(RegTest.get_score(rtn_pred, rtn_true))
        return pd.DataFrame(score, index=self.date[2:-2])


reg = RegTest(df, {"BETA": beta, "MOM": mom, "VOL": vol, "PVCOV": pvcov,
                   "REVERSE": reverse, "VOLSPIKE": volspike}, "OLS",
              start="2009-03-01", end="2011-12-31")


#%% 4. backtest system

class DailyBackTest:
    def __init__(self, td_pos, td_univ):
        self.td_pos = td_pos.copy()
        self.td_univ = td_univ.copy()
    
    def update(self, date, capital, td_alpha, td_univ, td_vol, td_std,
               dl_univ):
        self.date = date
        self.capital = capital
        self.cur_pos = self.td_pos.copy().rename(columns={"td_pos": "cur_pos"})
        self.cur_univ = self.td_univ.copy().rename(columns={"td_uni":
                                                            "cur_uni"})
        self.td_alpha = td_alpha.copy()      # today's factor score
        self.td_univ = td_univ.copy()        # today's universe
        self.td_vol = td_vol.copy()          # today's volume
        self.td_std = td_std.copy()          # today's volatility
        self.dl_univ = dl_univ.copy()        # today's delist stocks
        
        self.td_profit = 0
        self.td_pos = pd.DataFrame()
        self.td_exe_price = pd.DataFrame()
        self.td_chg_pos = pd.DataFrame()
    
    # ? td_univ has nan
    def get_profit(self):
        temp = pd.concat([self.cur_pos, self.cur_univ, self.td_univ], axis=1)
        #  in cur_univ but not in td_univ: suspend
        #  ====> cover without profit or cost
        temp.iloc[:,2] = temp.iloc[:,2].fillna(temp.iloc[:,1])
        #  profit[t] = pos[t-1] * (price[t] - price[t-1]) (- other costs)
        temp["td_profit"] = temp.iloc[:,0] * (temp.iloc[:,2] - temp.iloc[:,1])
        self.td_profit = temp["td_profit"].sum()
        return
    
    def get_pos(self):
        #  w[i] = factor[i] / sum(abs(factor[i]))
        temp = (self.td_alpha.iloc[:,0] /
                abs(self.td_alpha.iloc[:,0]).sum()).to_frame()
        temp = pd.concat([self.cur_univ, self.dl_univ, temp], axis=1)
        #  pos[i] = Int(capital * w[i] / price[i]) if not delist else 0
        temp["td_pos"] = self.capital * temp.iloc[:,2] / temp.iloc[:,0]
        temp["td_pos"] = temp["td_pos"].fillna(0).apply(round) * temp.iloc[:,1]
        self.td_pos = temp[["td_pos"]]
        return
    
    def get_exe_price(self):
        #  develop your own execution strategy here
        #  now it is set to be executed with close price
        self.td_exe_price = self.td_univ
        return
    
    def get_trading_cost(self):
        #  develop your own trading cost model here
        #  now f(q) = σ * √(|q|/volume), cost = f(q) * |q|
        temp = pd.concat([self.td_vol, self.td_std, self.td_chg_pos], axis=1)
        cost = (temp.iloc[:,1] * np.sqrt(abs(temp.iloc[:,2]) /
                                         temp.iloc[:,0]) *
                abs(temp.iloc[:,2])).sum()
        return cost
    
    def get_change_pos(self):
        temp = pd.concat([self.cur_pos, self.td_pos, self.td_exe_price,
                          self.td_univ], axis=1)
        temp.iloc[:,0] = temp.iloc[:,0].fillna(0)
        temp.iloc[:,1] = temp.iloc[:,1].fillna(0)
        temp["chg_pos"] = temp.iloc[:,1] - temp.iloc[:,0]
        self.td_chg_pos = temp[["chg_pos"]]
        cost = self.get_trading_cost()
        #  executed price should be use to calculate profit instead of close
        loss = ((temp.iloc[:,3] - temp.iloc[:,2]) *
                self.td_chg_pos.iloc[:,0]).sum()
        self.td_profit -= (loss + cost)
        return
    
    def execution(self):
        self.get_profit()
        self.get_pos()
        self.get_exe_price()
        self.get_change_pos()
        return


class BackTest:
    def __init__(self, title, df, factor, model="OLS", merge=None,
                 long=False, pct=0, capital=1000000, start=None, end=None):
        self.title = title
        self.df = df
        self.factor = factor
        self.model = model
        self.merge = merge
        self.long = long
        self.pct = pct
        self.total_cap = capital
        self.capital = capital
        
        self.univ = (self.df["PRC"] / self.df["CFACPR"]).unstack().loc[start:
                                                                       end]
        self.rtn = self.univ.pct_change()
        self.vol = self.df["VOL"].unstack().loc[start:end]
        self.std = self.df["STD"].unstack().loc[start:end]
        self.dl_univ = self.df["DLPRC"].apply(lambda x: 1 if np.isnan(x) else
                                              0).unstack().loc[start:end]
        
        self.mkt_rtn = self.rtn.mean(axis=1)
        self.mkt_rtn = self.mkt_rtn.iloc[1:]
        self.mkt_rtn.iloc[0] = 0
        
        self.date = self.univ.index.values
        self.profit = [0, 0]
        self.pnl = [capital, capital]
        self.pnl_list = {}
    
    def update(self, title=None, factor=None, model=None, merge=None,
               long=None, pct=None):
        self.title = self.title if title is None else title
        self.factor = self.factor if factor is None else factor
        self.model = self.model if model is None else model
        self.merge = self.merge if merge is None else merge
        self.long = self.long if long is None else long
        self.pct = self.pct if pct is None else pct
        self.capital = self.total_cap
        self.profit = [0, 0]
        self.pnl = [self.capital, self.capital]
    
    def get_X(self, idx):
        X = pd.concat([self.rtn.loc[self.date[idx]]] +
                      [self.factor[i].loc[self.date[idx-2]] for i in
                       self.factor], axis=1)
        X.columns = ["rtn"] + list(self.factor.keys())
        return X
    
    def get_td_alpha(self, idx):
        td_alpha =  pd.concat([self.factor[i].loc[self.date[idx]] for i in
                               self.factor], axis=1)
        td_alpha.columns = list(self.factor.keys())
        return td_alpha
    
    def execution(self):
        td_univ = self.univ.loc[self.date[0]].rename("td_univ").to_frame()
        td_pos = td_univ.copy()
        td_pos.iloc[:,0] = 0
        td_pos.rename(columns={"td_univ": "td_pos"}, inplace=True)
        
        td_bt = DailyBackTest(td_pos, td_univ)
        td_reg = DailyReg()
        
        n = len(self.date)
        for i in range(2, n-2):
            X = self.get_X(i)
            td_alpha = self.get_td_alpha(i)
            td_reg.update(X, td_alpha)
            alpha = td_reg.get_reg(self.model, self.merge)
            if self.long:  # long only
                alpha = td_reg.LONG(alpha,
                                    self.pct).rename("td_alpha").to_frame()
            else:  # long-short balance
                alpha = td_reg.LSB(alpha,
                                   self.pct).rename("td_alpha").to_frame()
            alpha["td_alpha"] = alpha["td_alpha"].fillna(0)
            
            date = self.date[i+1]
            td_bt.update(date, self.capital, alpha,
                         self.univ.loc[date].rename("td_uni").to_frame(),
                         self.vol.loc[date].rename("td_vol").to_frame(),
                         self.std.loc[date].rename("td_std").to_frame(),
                         self.dl_univ.loc[date].rename("dl_uni").to_frame())
            td_bt.execution()
            self.profit.append(td_bt.td_profit)
            self.capital += td_bt.td_profit
            self.pnl.append(self.capital)
        
        alpha.iloc[:,0] = 0  # cover positions
        date = self.date[i+2]
        td_bt.update(date, self.capital, alpha,
                     self.univ.loc[date].rename("td_uni").to_frame(),
                     self.vol.loc[date].rename("td_vol").to_frame(),
                     self.std.loc[date].rename("td_std").to_frame(),
                     self.dl_univ.loc[date].rename("dl_uni").to_frame())
        td_bt.execution()
        self.profit.append(td_bt.td_profit)
        self.capital += td_bt.td_profit
        self.pnl.append(self.capital)
        self.pnl_list.update({self.title: self.pnl})
        return
    
    def single_alpha_test(self):
        title = self.title
        factor_list = self.factor.copy()
        for i in factor_list:
            self.update(title=i, factor={i: factor_list[i]})
            self.execution()
        self.factor = factor_list
        self.title = title
        return
    
    def single_model_test(self, model_list):
        title = self.title
        temp = self.model
        for i in model_list:
            self.update(title=i, model=i)
            self.execution()
        self.model = temp
        self.title = title
        return
    
    @staticmethod
    def statistic(pnl):
        stat = {}
        stat["tot_rtn"] = pnl.iloc[-1] / pnl.iloc[0] - 1
        stat["ann_rtn"] = stat["tot_rtn"] / len(pnl) * 252
        stat["ann_std"] = pnl.pct_change().std() * np.sqrt(252)
        stat["Sharpe"] = stat["ann_rtn"] / stat["ann_std"]
        stat["MDD"] = max(np.maximum.accumulate(pnl) - pnl)
        stat["MDD (%)"] = max(1 - pnl / np.maximum.accumulate(pnl))
        return pd.Series(stat)
    
    @staticmethod
    def summary(pnl):
        tot = pnl.apply(BackTest.statistic)
        pnl["year"] = [i.year for i in pnl.index]
        ann_rtn = pnl.groupby("year").apply(lambda x: x.iloc[-1] /
                                            x.iloc[0] - 1).iloc[:,:-1]
        ann_std = pnl.groupby("year").apply(lambda x: x.pct_change().std() *
                                            np.sqrt(252)).iloc[:,:-1]
        ann_Sharpe = ann_rtn / ann_std
        return tot, ann_rtn, ann_Sharpe
    
    def get_pnl(self, hedge=False):
        pnl = pd.DataFrame(self.pnl_list, index=self.date[1:])
        if hedge:
            rtn = pnl.pct_change()
            rtn["MKT"] = self.mkt_rtn
            for i in pnl.columns:
                rtn[i] = self.total_cap * ((rtn[i] - rtn["MKT"]).fillna(0) +
                                           1).cumprod()
            return rtn
        return pnl
    
    def plot(self, plist=None, market=False, hedge=False, corr=False):
        plt.figure(figsize=(12, 4))
        plist = list(self.pnl_list.keys()) if plist is None else plist
        for i in plist:
            plt.plot(self.date[1:], self.pnl_list[i], label=i)
        if market:
            mkt_rtn = self.total_cap * (self.mkt_rtn + 1).cumprod()
            plt.plot(self.date[1:], mkt_rtn, label="MKT")
        plt.legend(loc="upper left")
        plt.show()
        
        if hedge:
            rtn = self.get_pnl(True)
            plt.figure(figsize=(12, 4))
            for i in plist:
                plt.plot(self.date[1:], rtn[i], label=i)
            plt.legend(loc="upper left")
            plt.show()
        
        if corr:
            plt.figure(figsize=(6, 6))
            if hedge:
                sns.heatmap(rtn[plist].corr(), annot=True, vmax=1, square=True,
                            cmap="Blues")
            else:
                sns.heatmap(pd.DataFrame(self.pnl_list)[plist].corr(),
                            annot=True, vmax=1, square=True, cmap="Blues")
            plt.show()
        return


#%% Factor Selection

bt = BackTest("ALL", df, {"BETA": beta, "MOM": mom, "VOL": vol, "PVCOV": pvcov,
                          "REVERSE": reverse, "VOLSPIKE": volspike},
              long=True, pct=0.5, start="2009-03-01", end="2011-12-31")
bt.single_alpha_test()
bt.plot(market=True, hedge=True, corr=True)


#%%

bt.update()
bt.execution()
bt.update("Factor_0", {"BETA": beta, "MOM": mom, "VOL": vol, "PVCOV": pvcov,
                       "REVERSE": reverse})
bt.execution()
bt.update("Factor_1", {"BETA": beta, "VOL": vol, "PVCOV": pvcov,
                       "REVERSE": reverse, "VOLSPIKE": volspike})
bt.execution()
bt.update("Factor_2", {"BETA": beta, "VOL": vol, "PVCOV": pvcov,
                       "REVERSE": reverse})
bt.execution()
bt.plot(plist=["ALL", "Factor_0", "Factor_1", "Factor_2"],
        market=True, hedge=True)


#%% Model Selection

bt = BackTest("ALL", df, {"BETA": beta, "MOM": mom, "VOL": vol, "PVCOV": pvcov,
                          "REVERSE": reverse},
              long=True, pct=0.5, start="2009-03-01", end="2011-12-31")
bt.single_model_test(["OLS", "Ridge", "RF", "GBDT"])
bt.plot(market=True, hedge=True, corr=True)


#%%

bt.update(model="Merge")
bt.execution()
bt.update("Model_0", model="Merge", merge=["OLS", "RF"])
bt.execution()
bt.update("Model_1", model="RankMerge", merge=["Combo", "OLS", "RF"])
bt.execution()
bt.update("Combo", model="Combo")
bt.execution()
bt.plot(plist=["ALL", "Model_0", "Model_1", "Combo"], market=True, hedge=True)


#%% out-of-sample test

bt = BackTest("Top50", df, {"BETA": beta, "MOM": mom, "VOL": vol, "PVCOV": pvcov,
                          "REVERSE": reverse},
              model="RankMerge", merge=["Combo", "OLS", "RF"],
              long=True, pct=0.5, start="2012-01-01", end="2012-12-31")
bt.execution()
bt.update("Top25", pct=0.25)
bt.execution()
bt.update("Top10", pct=0.1)
bt.execution()
bt.plot(plist=["Top50", "Top25", "Top10"], market=True, hedge=True)