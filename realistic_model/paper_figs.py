# Paper figures: model fit to reported cases, Figures 3a-c (uniform) and 4a-c (income-based), and the strategy comparison.
# Plain matplotlib style to match the other figures in the paper. Run after fit.py and scenarios.py.
import sys, pickle
exec(open(sys.argv[1]).read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[2]
saved = pickle.load(open(f'{SCR}/{FIT_FILE}', 'rb')); rr = tractFactors(saved['zf'])
VSTART, MID = 286, 397   # rollout starts Dec 11 2020; vaccination map on Apr 1 2021
dates = DAY0 + pd.to_timedelta(np.arange(END + 1), 'D')

def runDaily(alloc):
  y = initial(saved['seed'], rr); totals = [y.sum(1)]; yMid = yStart = None
  for ws, we, b in saved['windows']:
    for day in range(ws, we):
      if day == VSTART: yStart = y.copy()
      if day == MID: yMid = y.copy()
      y, _ = step_days(y, day, day + 1, b, rr, alloc); totals.append(y.sum(1))
  return np.array(totals), yStart, yMid, y

geo = geopd.read_file('nycCensusTracts/nyct2020.shp'); geo['GEOID'] = geo['GEOID'].astype(np.int64)
geo = geo[['GEOID', 'geometry']].merge(df[['GEOID']], on='GEOID')   # same tract order as df
assert (geo.GEOID.to_numpy() == df.GEOID.to_numpy()).all()

res = {}
for name, alloc in [('uniform', uniformAllocation), ('income_based', incomeAllocation)]:
  totals, yStart, yMid, yEnd = runDaily(alloc)
  res[name] = dict(totals=totals, vaxMid=(yMid[1] + yMid[3] + yMid[5] + yMid[7]) / pop,
                   infRollout=(yEnd[8] - yStart[8]) / pop)
  print(name, f'infections during rollout {(yEnd[8] - yStart[8]).sum():.0f}, recovered at end {(yEnd[6] + yEnd[7]).sum():.0f}, '
        f'vaccinated at end {(yEnd[1] + yEnd[3] + yEnd[5] + yEnd[7]).sum():.0f}')

TITLES = {'uniform': 'Uniform Allocation', 'income_based': 'Income-Based Allocation'}
TITLE, LABEL = 14, 14

# model fit: reported cases, citywide and by borough
rep = saved['onsets'] * np.array([rho(dd, rhoEarly) for dd in range(END)])[:, None]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates[:END], citySmooth, label='Observed (7-day average)')
ax.plot(dates[:END], rep.sum(1), label='Model')
ax.set_title('Reported COVID-19 Cases in NYC: Model vs Observed', fontsize=TITLE)
ax.set_xlabel('Date', fontsize=LABEL); ax.set_ylabel('Reported Cases per Day', fontsize=LABEL)
ax.grid(True); ax.legend()
fig.tight_layout(); fig.savefig(f'{OUT}/model_fit.png', dpi=150); plt.close(fig)

fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
for ax, b in zip(axes, BOROS):
  n = pop[boro == b].sum() / 1e5
  ax.plot(dates[:END], pd.Series(boroCases[b]).rolling(7, center=True, min_periods=1).mean() / n, label='Observed (7-day average)')
  ax.plot(dates[:END], rep[:, boro == b].sum(1) / n, label='Model')
  ax.set_title(b, fontsize=12); ax.grid(True)
axes[0].legend(); axes[2].set_ylabel('Reported Cases per Day per 100,000', fontsize=LABEL); axes[-1].set_xlabel('Date', fontsize=LABEL)
fig.suptitle('Reported COVID-19 Cases by Borough: Model vs Observed', fontsize=TITLE)
fig.tight_layout(); fig.savefig(f'{OUT}/model_fit_borough.png', dpi=150); plt.close(fig)

# (a) citywide compartments over time
SERIES = [('Susceptible', lambda T: T[:, 0]), ('Exposed', lambda T: T[:, 2] + T[:, 3]), ('Infected', lambda T: T[:, 4] + T[:, 5]),
          ('Recovered', lambda T: T[:, 6] + T[:, 7]), ('Vaccinated, never infected', lambda T: T[:, 1])]
for name in res:
  T = res[name]['totals'] / 1e6
  fig, ax = plt.subplots(figsize=(10, 6))
  for label, f in SERIES: ax.plot(dates, f(T), label=label)
  ax.axvline(dates[VSTART], color='gray', ls='--', label='Rollout starts')
  ax.set_title(f'SEIRV Model Simulation - {TITLES[name]}', fontsize=TITLE)
  ax.set_xlabel('Date', fontsize=LABEL); ax.set_ylabel('Number of People (millions)', fontsize=LABEL)
  ax.set_xlim(dates[0] - pd.Timedelta(days=15), dates[-1] + pd.Timedelta(days=15)); ax.grid(True); ax.legend()
  fig.tight_layout(); fig.savefig(f'{OUT}/new_seirv_{name}.png', dpi=150); plt.close(fig)

# (b) % with a first dose by tract on Apr 1 2021, (c) % infected during the rollout; one color scale per figure pair
for key, title, cbar, fname in [('vaxMid', 'Percentage of People with a First Dose by Tract (April 1, 2021)', 'Percentage of Population Vaccinated', 'new_vaccinated_apr2021'),
                                ('infRollout', 'Percentage of People Infected During the Rollout by Tract (Dec 2020 - Nov 2021)', 'Percentage of Population Infected', 'new_infected_rollout')]:
  vmin, vmax = np.percentile(np.concatenate([100 * res[k][key] for k in res]), [2, 98])
  for name in res:
    g = geo.copy(); g['v'] = 100 * res[name][key]
    fig, ax = plt.subplots(figsize=(8, 7))
    g.plot(column='v', cmap='viridis', vmin=vmin, vmax=vmax, ax=ax, linewidth=0, legend=True, legend_kwds={'label': cbar})
    ax.set_axis_off(); ax.set_title(f'{title}\n{TITLES[name]}', fontsize=12)
    fig.savefig(f'{OUT}/{fname}_{name}.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# strategy comparison: share of each income quartile infected during the rollout
scen = pickle.load(open(f'{SCR}/scenarios2_rho{RHO_LATE}_cap.pkl', 'rb'))
q = pd.qcut(inc, 4, labels=False); QN = ['Poorest', 'Q2', 'Q3', 'Richest']
names = [k for k in scen if k != 'No vaccine']
fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(4); w = 0.8 / len(names)
for k, name in enumerate(names):
  inf = scen[name]['onsets'][VSTART:].sum(0)
  ax.bar(x + (k - (len(names) - 1) / 2) * w, [100 * inf[q == j].sum() / pop[q == j].sum() for j in range(4)], w, label=name)
ax.set_xticks(x, QN)
ax.set_title('Percentage of People Infected During the Rollout by Income Quartile', fontsize=TITLE)
ax.set_xlabel('Census Tract Income Quartile', fontsize=LABEL); ax.set_ylabel('Percentage Infected', fontsize=LABEL)
ax.set_ylim(0, 27); ax.grid(True, axis='y'); ax.set_axisbelow(True); ax.legend(ncol=3, loc='upper center')
fig.tight_layout(); fig.savefig(f'{OUT}/strategies_by_quartile.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5)); d = dates[1:]
for name in names: ax.plot(d[VSTART - 30:], scen[name]['onsets'][VSTART - 30:].sum(1), label=name)
ax.set_title('New Infections per Day During the Rollout by Strategy', fontsize=TITLE)
ax.set_xlabel('Date', fontsize=LABEL); ax.set_ylabel('New Infections per Day', fontsize=LABEL)
ax.grid(True); ax.legend()
fig.tight_layout(); fig.savefig(f'{OUT}/strategies_over_time.png', dpi=150); plt.close(fig)
