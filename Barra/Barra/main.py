import polars as pl
from datetime import datetime
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from src.loader import Loader

class Processor:
    def __init__(self):
        self.basepath = Path.cwd() / "data" 
        self.basepath.mkdir(exist_ok=True, parents=True)
        self.identifiers = ["symbol", "date"]

    def plot_null_heatmap(self, df):
        null_mask = df.select(pl.all().is_null()).to_pandas()
        plt.figure(figsize=(12, 8))
        sns.heatmap(null_mask, cbar=False, cmap="viridis")
        plt.title("Missing Data Heatmap (Yellow = Missing, Purple = Present)")
        plt.show()

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
        
    def neutralise_factors(self, df, factors, risk_factors, schema):
        #F= a + BX + e
        #,where F is the raw factor signal, X is exposures risk factors, B is the regression coefficients
        #,a to fit cross-sectional mean so that residual mu = 0 and and e is the 'pure' factor (X^T @ e = 0)
        
        required_cols = factors + risk_factors #(sector, beta, size, vol, etc)
        total_dates = df.select(pl.col("date").n_unique()).collect().item()
        print(f"[INFO]Running cross-sectional regressions across {total_dates:,} dates: ")
        print(f"[INFO]{len(factors)} factors against {len(risk_factors)} risk factors")
        df = df.with_columns(
            pl.col(f).fill_null(pl.col(f).mean().over(risk_factors)) ##temp fix
            for f in required_cols
        )
        
        def neutralise_group(group_df):
            N = group_df.height
            
            intercept = np.ones(N) 
            dummies = group_df.select(dummy_cols).to_numpy() 
            X = np.column_stack([intercept, dummies]) #(N, K) ##Add risk factors here
            Y = group_df.select(factors).to_numpy() #(N, F)
            K = X.shape[1]
            
            #Ridge Regression: (X'X + λI) β = X'Y
            l2_lambda = 0.1 #λI applies penalty to large β
            LHS = X.T @ X + l2_lambda * np.eye(K) #(K, K)
            RHS = X.T @ Y #(K, F)
            beta = np.linalg.solve(LHS, RHS) #Solves for β (K, F)
            residuals = Y - X @ beta #(N, F)
            return group_df.drop(dummy_cols).with_columns(
                [pl.Series(f, residuals[:, i]) for i, f in enumerate(factors)] #(N,) x F -> (N, F)
            )
        return df.group_by("date", maintain_order=True).map_groups(neutralise_group, schema) #(N, F) -> (N, F)
    
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

    def process_factors(self, df, factors, composite):
        return (
            df
            #Components
            .pipe(self.winsor_factors, factors, p=0.01) 
            .pipe(self.znorm_factors, factors)
            .pipe(self.combine_factors, factors, composite)
        )
    def process_composites(self, df, factors, risk_factors, schema):
        return (
            df.pipe(self.znorm_factors, factors)
            .pipe(self.neutralise_factors, factors, risk_factors, schema)
            .pipe(self.rescale_factors, factors)
        )

#-----//Params//-----
symbols = ["AAPL", "META"]#, "MSFT", "WMT", "COST", "NVDA", "TSLA", "AVGO", "NFLX", "LLY", "JNJ", "AMD", "RBLX", "BMBL", "ABVX", "PDD", "CHA", "SRPT", \
           #"IONQ", "QUBT", "QBTS", "BABA", "BIDU", "NVO", "UNH", "MHO", "HMC", "GS", "MS", "JPM", "BAC", "C", "AXP", "COIN", "AEO", "GAP", "CAL", "SCVL", \
           #"OKLO", "PLTR", "HIMS"] #use sampling
factor_defs = {
    "MOM": [["UMD_12_1", "UMD_6_1", "UMD_3_1"], 21, 1], #skips to ignore STR
    "VAL": [["HML_3", "HML_5"], 21*12, -1]
}
#21 trading days per month

pf_schema = {
    "symbol": pl.Utf8,
    "country": pl.Utf8,
    "industry": pl.Utf8,
    "sector": pl.Utf8,
    "ts": pl.Utf8
}

his_schema = {
    "symbol": pl.Utf8,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "adjclose": pl.Float64,
    "splits": pl.Float64,
    "ts": pl.Utf8
}

composite_schema = {
    "symbol": pl.Utf8,
    "date": pl.Date,
    "MOM": pl.Float64,
    "VAL": pl.Float64,
    "country": pl.Utf8,
    "industry": pl.Utf8,
    "sector": pl.Utf8
}

categories = [c for c in pf_schema.keys() if c not in ["symbol", "ts"]]
composite_factors = list(factor_defs.keys())


#-----//Loading//-----
loader = Loader()
processor = Processor()

loader.compact_data("Profile", pf_schema)
loader.compact_data("History", his_schema)

lf = loader.load_data(symbols, "Profile", fetch_func=loader.fetch_profile, schema=pf_schema)
pf = lf.sort("ts", descending=False).unique(subset=["symbol"], keep="last").drop("ts") 

lf = loader.load_data(symbols, "History", fetch_func=loader.fetch_history, other_args={"period": "max"}, schema=his_schema)
his = lf.sort("ts", descending=False).unique(subset=["symbol", "date"], keep="last")
his = his.drop("ts").drop(['open', 'high', 'low', 'volume', 'close', 'splits'])


#-----//Processing//-----
lf = processor.log_transform(his.collect()) 

#Factor Construction
for factors, unit, k in factor_defs.values():
    for factor in factors:
        val = factor.split("_") + [0]
        lf = processor.add_log_change(factor, lf, int(val[1])*unit, int(val[2])*unit, k=k)
for composite, (factors, _, _) in factor_defs.items():
    lf = processor.process_factors(lf, factors, composite)

#One Hot Encoding
pf = pf.with_columns([pl.col(cat).fill_null("Unknown").alias(cat)
    for cat in categories
]).sort(categories)
df = lf.join(pf, on="symbol", how="left").collect()
dummy_df = df.select(categories).to_dummies(drop_first=True) #drop_first=True so no perfect multicollinearity
dummy_cols = dummy_df.columns
df = pl.concat([df, dummy_df], how="horizontal")

risk_factors = dummy_cols
lf = df.lazy().select(["symbol", "date"]+composite_factors+risk_factors)
#processor.plot_null_heatmap(df)
lf = processor.process_composites(lf, composite_factors, risk_factors, composite_schema).sort(["symbol", "date"])

print("")
df = lf.select(["symbol", "date"]+composite_factors).filter(
    pl.col(f).is_not_null() & pl.col(f).is_not_nan()
    for f in composite_factors
).collect()
df.write_csv("test.csv")
print(df.tail())
print(df.shape)


#unit test, smooth forward fill




