import pandas as pd, numpy as np, lightgbm as lgb, model
from sklearn.metrics import root_mean_squared_error

mov = model.load_movements()
dep = mov[mov['PHASE_mvt']=='DEP'].copy()          # NO artifact filter (keep them for eval)
pool = model.build_congestion_pool(mov)
weather = model.load_weather_cache()
cong = model.compute_congestion_signal(dep, pool, model.CONGESTION_WINDOW_MINUTES)
cs   = model.compute_congestion_signal(dep, pool, model.CONGESTION_WINDOW_SHORT_MINUTES)
dayd = model.compute_day_deviation_ratio(dep, pool)
feats = model.build_features(dep, cong, dayd, cs-cong, weather)
tgt = dep['TAXITIME_SEC_mvt'].astype(float)
mo = dep['MVT_TIME_UTC_mvt'].dt.month.values
cats = list(model.CATEGORICAL_FEATURES)
params = dict(objective='regression', metric='rmse', num_leaves=127, learning_rate=0.05,
              min_data_in_leaf=50, feature_fraction=0.8, seed=42, verbose=-1)

print('%-6s %8s %8s %8s %6s'%('evalMo','clean','HONEST','n_art','artSSE%'))
rows=[]
for M in [4,5,6,7,10,12]:
    tr = (mo < M) & (tgt.values <= 21600)          # train: prior months, filtered
    va = (mo == M)                                  # eval: this month, artifacts INCLUDED
    if tr.sum()<1000 or va.sum()==0: continue
    ds = lgb.Dataset(feats[tr], label=tgt[tr], categorical_feature=cats, free_raw_data=False)
    b = lgb.train(params, ds, num_boost_round=300)
    p = b.predict(feats[va]); y = tgt[va].values
    art = y>21600
    clean = root_mean_squared_error(y[~art], p[~art])
    honest = root_mean_squared_error(y, p)
    sse=(p-y)**2
    artpct = 100*sse[art].sum()/sse.sum() if art.sum()>0 else 0
    print('%-6d %8.1f %8.1f %8d %6.0f'%(M, clean, honest, art.sum(), artpct))
    rows.append((va.sum(), honest))
# pooled honest RMSE across the ranking-relevant months (Apr-Jul), weighted by rows
sel=[r for r,M in zip(rows,[4,5,6,7,10,12])]
