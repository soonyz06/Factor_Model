from yahooquery import Ticker
import polars as pl
from datetime import datetime
from pathlib import Path
import re
from functools import reduce
import time


class Loader:
    def __init__(self):
        self.basepath = Path.cwd() / "data" 
        self.basepath.mkdir(exist_ok=True, parents=True)
        self.identifiers = ["symbol", "date"]
         
    def fetch_history(self, symbols, period, interval="1d"): 
        assert isinstance(symbols, list), "Symbols should be a list"
        if len(symbols)==0: return None
        
        print(f"Fetching History ({period},{interval}) -> {len(symbols)}")
        ticker = Ticker(symbols, asynchronous=True)
        history = ticker.history(period=period, interval=interval)
        if history.empty:
            print("Dataframe is empty")
            return None
        history = pl.from_pandas(history, include_index=True)
        history = history.drop("dividends")
        history = history.with_columns(pl.col("date").cast(pl.Date))
        return history

    def fetch_profile(self, symbols):
        assert isinstance(symbols, list), "Symbols should be a list"
        if len(symbols)==0: return None
        print(f"Fetching Profile -> {len(symbols)}")
        ticker = Ticker(symbols, asynchronous=True)
        profile = pl.DataFrame({"symbol": key, **val} for key, val in ticker.summary_profile.items()).select(["symbol", "country", "industry", "sector"])
        return profile
        
    def load_data(self, symbols, dirname, fetch_func, other_args=None): #add batching + updating + compact 
        if other_args is None: other_args = {}
        dirpath = self.basepath / dirname
        dirpath.mkdir(exist_ok=True, parents=True)
        existing_symbols = set()

        if list(dirpath.glob("*.parquet")):
            try:
                lf = (pl.scan_parquet(dirpath / "*.parquet").select("symbol").unique().collect())
                existing_symbols = set(lf["symbol"])
            except (FileNotFoundError, pl.exceptions.ComputeError):
                pass
            except Exception as e:
                print(f"[ERROR] Failed to load files due to: {e}")

        missing_symbols = list(set(symbols) - existing_symbols)
        if missing_symbols:
            #print(f"Fetching missing symbols -> {len(missing_symbols)}")
            missing_df = fetch_func(missing_symbols, **other_args)

            if missing_df is not None and not missing_df.is_empty():
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                missing_df = missing_df.with_columns(
                    pl.lit(ts).alias("ts")
                )
                missing_df.write_parquet(dirpath / f"{ts}.parquet")

        if not list(dirpath.glob("*.parquet")):
            print("[ERROR] Failed load.")
            return None
    
        lf =  pl.scan_parquet(dirpath / "*.parquet")
        return lf

class Processor:
    def __init__(self):
        self.basepath = Path.cwd() / "data" 
        self.basepath.mkdir(exist_ok=True, parents=True)
        self.identifiers = ["symbol", "date"]

    def log_transform(self, df):
        #Chose to use log ret instead of raw ret due to the following:
        #ln(a/b) = ln(a)-ln(b) ~ additive 
        #ln(20/10) + ln(5/10) = 0 ~ symmetry
        
        return (
            df.sort(self.identifiers).upsample(time_column="date", every="1d", group_by="symbol") #ensures trading day alignment
            .with_columns(
                pl.col("adjclose").forward_fill().over("symbol") 
            )
            .lazy() 
            .filter(pl.col("date").dt.weekday()<=5) 
            .with_columns(
                log_ret = pl.col("adjclose").log().diff().over("symbol").fill_null(0)
            )
        )

    def add_log_change(self, factor, df, lookback_days, gap_days=0, k=1):        
        min_obs =  0.8
        window_days = int(lookback_days - gap_days)
        
        return df.with_columns(
            (pl.col("log_ret")
             .shift(gap_days) #pushed forward in time 
             .rolling_sum( 
                 window_size=window_days,
                 weights=None, 
                 min_samples=int(window_days*min_obs) #min_obs validation
             )
             * k
            )
            .over("symbol")
            .alias(factor)            
        )
                   
    def winsor_factors(self, df, factors, p=0.01):
        return df.with_columns([
            pl.col(f).clip(
                pl.col(f).quantile(p).over("date"),
                pl.col(f).quantile(1-p).over("date")
            )
            .alias(f)
            for f in factors
        ])

    def combine_factors(self, df, factors, composite):
       return df.with_columns(
               pl.mean_horizontal(factors).alias(composite) #automatic reweighting (0.5*A + 0.5*B -> 1*A + 0*B when B is null), could also do sum_horizontal / used_weights
        ).drop(factors)

    def znorm_factors(self, df, factors):
        return df.with_columns([
            (pl.col(f) - pl.col(f).mean().over("date")) / (pl.col(f).std().over("date") + + 1e-8)
            .alias(f)
            for f in factors
        ])

    def neutralise_factors(self, df, factors):
        #F=BX+e, where F is raw factor, X is risk factors and B is estimated
        #keeps mu=0, so only need to rescale sd
        return df.with_columns( #PLACEHOLDER -> equivalent to OLS of categorical
            (pl.col(f) - pl.col(f).mean().over(["date", "sector"]))
            .alias(f)
            for f in factors
        )
    
    def rescale_factors(self, df, factors):
        return df.with_columns([
            pl.col(f) / (pl.col(f).std().over("date") + + 1e-8)
            .alias(f)
            for f in factors
        ])        

    def reverse_winsor(self, df, factor, p=0.05):
        low_mask = pl.col(factor).quantile(p).over("date")
        high_mask = pl.col(factor).quantile(1-p).over("date")
        
        return df.with_columns(
            pl.when(pl.col(factor).is_between(low_mask, high_mask, closed="none")) #exclusive
            .then(pl.lit(None))
            .otherwise(pl.col(factor))
            .alias(factor)
        )

    def process_factor(self, df, factors, composite):
        return (
            df
            #Components
            .pipe(self.winsor_factors, factors, p=0.01) 
            .pipe(self.znorm_factors, factors)
            .pipe(self.combine_factors, factors, composite)
            .pipe(self.znorm_factors, [composite])

            #Composite z-normalised signal
            .pipe(self.neutralise_factors, [composite])
            .pipe(self.rescale_factors, [composite])
        )

symbols = ["AAPL", "META", "MSFT", "WMT", "COST", "NVDA"]
factor_defs = {
    "MOM": [["UMD_12_1", "UMD_6_1", "UMD_3_1"], 21, 1], #skips to ignore STR
    "VAL": [["HML_3", "HML_5"], 21*12, -1]
}
#21 trading days per month

#OLS equation: \hat{\beta} = (X'X)^{-1}X'y
loader = Loader()
processor = Processor()


#Loading
lf = loader.load_data(symbols, "Profile", fetch_func=loader.fetch_profile) 
pf = lf.unique(subset=["symbol"], keep="last").drop("ts") #assuming read in order
#lf = lf.sort("ts", descending=False).group_by(["symbol", "date"]).last()

lf = loader.load_data(symbols, "History", fetch_func=loader.fetch_history, other_args={"period": "max"})
his = lf.unique(subset=["symbol", "date"], keep="last").drop("ts").drop(['open', 'high', 'low', 'volume', 'close', 'splits'])


#Processing
categories = pf.drop(["symbol"]).collect_schema().names()
pf = pf.with_columns([ #Label Encoding
    pl.col(cat)
    .fill_null("Unknown")
    .cast(pl.Categorical)
    .to_physical() 
    .alias(cat)
    for cat in categories
])
df = his.join(pf, on="symbol", how="left").collect()

lf = processor.log_transform(df) #returns lazy

for factors, unit, k in factor_defs.values():
    for factor in factors:
        val = factor.split("_") + [0]
        lf = processor.add_log_change(factor, lf, int(val[1])*unit, int(val[2])*unit, k=k)

for composite, (factors, _, _) in factor_defs.items():
    lf = processor.process_factor(lf, factors, composite)

df = lf.collect()
print(df.columns)
print(df.tail())


#unit test, smooth forward fill
#sampling




