import warnings
import pandas as pd
import numpy as np
from scripts.project_helper import Sector
from zipline.pipeline.factors import CustomFactor, DailyReturns, Returns,RSI, SimpleMovingAverage, AnnualizedVolatility,AverageDollarVolume
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")


def momentum_1yr(window_length, universe, sector):
    return Returns(window_length=window_length, mask=universe) \
        .demean(groupby=sector) \
        .rank() \
        .zscore()

def mean_reversion_5day_sector_neutral_smoothed(window_length, universe, sector):
    unsmoothed_factor = - Returns(window_length=window_length, mask=universe) \
        .demean(groupby=sector) \
        .rank() \
        .zscore()
    return SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=window_length) \
        .rank() \
        .zscore()

class CTO(Returns):
    """
    Computes the overnight return, per hypothesis from
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554010
    """
    inputs = [USEquityPricing.open, USEquityPricing.close]
    
    def compute(self, today, assets, out, opens, closes):
        """
        The opens and closes matrix is 2 rows x N assets, with the most recent at the bottom.
        As such, opens[-1] is the most recent open, and closes[0] is the earlier close
        """
        out[:] = (opens[-1] - closes[0]) / closes[0]
        
class TrailingOvernightReturns(Returns):
    """
    Sum of trailing 1m O/N returns
    """
    window_safe = True
    
    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)

def overnight_sentiment_smoothed(cto_window_length, trail_overnight_returns_window_length, universe):
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    unsmoothed_factor = TrailingOvernightReturns(inputs=[cto_out], window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()
    return SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()

def rsi_sector_neutral(window_length,universe,sector):
    returns = Returns(window_length=window_length, mask=universe) \
        .demean(groupby=sector) \
        .rank() \
        .zscore()
    return RSI(inputs=[returns],window_length = window_length).rank().zscore()

class RegressionAgainstTime(CustomFactor):

    # choose a window length that spans one year's worth of trading days
    window_length = 252
    
    # use USEquityPricing's close price
    inputs = [USEquityPricing.close]
    
    # set outputs to a list of strings, which are names of the outputs
    # We're calculating regression coefficients for two independent variables, 
    # called beta and gamma
    outputs = ['beta', 'gamma']
    
    def compute(self, today, assets, out, dependent):
        
        # of the window length. E.g. [1,2,3...252]
        t1 = np.arange(self.window_length)
        
        t2 = np.arange(self.window_length)**2
        
        # combine t1 and t2 into a 2D numpy array
        X = np.array([t1,t2]).T

    
        #the number of stocks is equal to the length of the "out" variable,
        # because the "out" variable has one element for each stock
        n_stocks = len(out)
        # loop over each asset

        for i in range(n_stocks):
            # "dependent" is a 2D numpy array that
            # has one stock series in each column,
            # and days are along the rows.
            # set y equal to all rows for column i of "dependent"
            y = dependent[:, i]
            
            # run a regression only if all values of y
            # are finite.
            if np.all(np.isfinite(y)):
                # create a LinearRegression object
                regressor = LinearRegression()
                
                # fit the regressor on X and y
                regressor.fit(X, y)
                
                # store the beta coefficient
                out.beta[i] = regressor.coef_[0]
                
                # store the gamma coefficient
                out.gamma[i] = regressor.coef_[1]
            else:
                # store beta as not-a-number
                out.beta[i] = np.nan
                
                # store gammas not-a-number
                out.gamma[i] = np.nan

class DownsideRisk(CustomFactor):
    '''Mean Returns divided by std of 1yr daily losses (Sortino Ratio)'''
    
    inputs = [USEquityPricing.close]
    
    window_length = 252
    
    def compute(self,today,assets,out,close):
        
        ret = pd.DataFrame(close).pct_change()
        
        out[:] = ret.mean().div(ret.where(ret < 0).std())
        
class Vol3M(CustomFactor):
    '''3-month Volatility: Standard Deviation of returns over 3 Months'''
    
    inputs = [USEquityPricing.close]
    window_length = 63
    
    def compute(self,today,assets,out,close):
        
        out[:] = np.log1p(pd.DataFrame(close).pct_change()).std()

class MarketDispersion(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        # returns are days in rows, assets across columns
        out[:] = np.sqrt(np.nanmean((returns - np.nanmean(returns))**2))

class MarketVolatility(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True
    
    def compute(self, today, assets, out, returns):
        mkt_returns = np.nanmean(returns, axis=1)
        out[:] = np.sqrt(260.* np.nanmean((mkt_returns-np.nanmean(mkt_returns))**2))

def compute_date_features(all_factors,start_date,end_date):
	
	all_factors['is_March'] = all_factors.index.get_level_values(0).month == 3
	
	all_factors['is_April'] = all_factors.index.get_level_values(0).month == 4
	
	all_factors['weekday'] = all_factors.index.get_level_values(0).weekday
	
	all_factors['quarter'] = all_factors.index.get_level_values(0).quarter
	
	all_factors['qtr_yr'] = all_factors.quarter.astype('str') + '_' + all_factors.index.get_level_values(0).year.astype('str')
	
	all_factors['month_end'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=start_date, end=end_date, freq='BM'))
	
	all_factors['month_start'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=start_date, end=end_date, freq='BMS'))
	
	all_factors['qtr_end'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=start_date, end=end_date, freq='BQ'))
	
	all_factors['qtr_start'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=start_date, end=end_date, freq='BQS'))

	return all_factors

def one_hot_encode_sectors(all_factors):
	
	nifty_500 = pd.read_csv('data/ind_nifty500list.csv')
	
	Industries = list(nifty_500['Industry'].unique())
	
	Industry_code = [i for i in range(1,len(Industries) + 1)]
	
	sector_lookup = dict(zip(Industry_code,Industries))

	sector_columns = []

	for sector_i, sector_name in sector_lookup.items():
		sector_column = 'sector_{}'.format(sector_name)
		sector_columns.append(sector_column)
		all_factors[sector_column] = (all_factors['sector_code'] == sector_i)

	return all_factors

def run_data_pipeline(engine,universe,start_date,end_date):
	
	pipeline = Pipeline(screen=universe)

	sector = Sector()

	# Alpha Factors :

	pipeline.add(DownsideRisk(),'Downside Risk (Sortino Ratio)')

	pipeline.add(Vol3M(),'3 Month Volatility')

	pipeline.add(momentum_1yr(252, universe, sector),'Momentum_1YR')

	pipeline.add(mean_reversion_5day_sector_neutral_smoothed(20, universe, sector),'Mean_Reversion_Sector_Neutral_Smoothed')

	pipeline.add(overnight_sentiment_smoothed(2, 10, universe),'Overnight_Sentiment_Smoothed')

	pipeline.add(rsi_sector_neutral(15,universe,sector),'RSI_Sector_Neutral_15d')

	pipeline.add(rsi_sector_neutral(30,universe,sector),'RSI_Sector_Neutral_30d')

	beta_factor = (RegressionAgainstTime(mask=universe).beta.rank().zscore())

	gamma_factor = (RegressionAgainstTime(mask=universe).gamma.rank().zscore())

	conditional_factor = (beta_factor*gamma_factor).rank().zscore()

	pipeline.add(beta_factor, 'time_beta')

	pipeline.add(gamma_factor, 'time_gamma')

	pipeline.add(conditional_factor, 'conditional_factor')

	# Universal Quant Features :

	pipeline.add(AnnualizedVolatility(window_length=20, mask=universe).rank().zscore(), 'volatility_20d')
	
	pipeline.add(AnnualizedVolatility(window_length=120, mask=universe).rank().zscore(), 'volatility_120d')
	
	pipeline.add(AverageDollarVolume(window_length=20, mask=universe).rank().zscore(), 'adv_20d')

	pipeline.add(AverageDollarVolume(window_length=120, mask=universe).rank().zscore(), 'adv_120d')

	pipeline.add(sector, 'sector_code')

	# Regime Features :

	pipeline.add(SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=20), 'dispersion_20d')

	pipeline.add(SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=120), 'dispersion_120d')

	pipeline.add(MarketVolatility(window_length=20), 'market_vol_20d')

	pipeline.add(MarketVolatility(window_length=120), 'market_vol_120d')

	# Target
	# Let's try to predict the go forward 1-week return. When doing this, it's important to quantize the target. The factor we create is the trailing 5-day return

	pipeline.add(Returns(window_length=5, mask=universe).quantiles(2), 'return_5d')

	pipeline.add(Returns(window_length=5, mask=universe).quantiles(25), 'return_5d_p')

	# Running the Pipeline

	all_factors = engine.run_pipeline(pipeline, start_date, end_date)

	# Computing Date Features

	all_factors = compute_date_features(all_factors, start_date, end_date)

	# One Hot Encoding Sectors

	all_factors = one_hot_encode_sectors(all_factors)

	# Shifted Target For Training The Model

	all_factors['target'] = all_factors.groupby(level=1)['return_5d'].shift(-5)

	return all_factors
