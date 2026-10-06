# New versions of paper Figures 3a-c (uniform) and 4a-c (income-based) from the fitted model.
import sys, pickle
exec(open(sys.argv[1]).read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
  print(name, f'infections during rollout {(yEnd[8] - yStart[8]).sum():.0f}')

TITLES = {'uniform': 'Uniform Allocation', 'income_based': 'Income-Based Allocation'}
SERIES = [('Susceptible', lambda T: T[:, 0], '#2a78d6'), ('Exposed', lambda T: T[:, 2] + T[:, 3], '#eb6834'),
          ('Infectious', lambda T: T[:, 4] + T[:, 5], '#1baf7a'), ('Recovered', lambda T: T[:, 6] + T[:, 7], '#eda100'),
          ('Vaccinated, never infected', lambda T: T[:, 1], '#e87ba4')]
INK, MUTED, GRID = '#1f1f1e', '#6b6b66', '#e6e6e3'

def style(ax):
  for s in ['top', 'right']: ax.spines[s].set_visible(False)
  for s in ['left', 'bottom']: ax.spines[s].set_color(MUTED)
  ax.tick_params(colors=MUTED); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

# (a) citywide compartments over time
ymax = max(res[k]['totals'][:, :8].sum(1).max() for k in res)
for name in res:
  T = res[name]['totals'] / 1e6
  fig, ax = plt.subplots(figsize=(10, 5.5)); style(ax)
  for label, f, col in SERIES:
    v = f(T); ax.plot(dates, v, color=col, lw=2, label=label)
    if label not in ('Exposed', 'Infectious'):   # these two end near zero and overlap; the legend names them
      ax.annotate(label, (dates[-1], v[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=INK)
  ax.axvline(dates[VSTART], color=MUTED, lw=1, ls='--'); ax.text(dates[VSTART], ymax / 1e6 * 0.98, ' Rollout starts', color=MUTED, fontsize=9, va='top')
  ax.set_ylim(0, ymax / 1e6 * 1.02); ax.set_xlim(dates[0], dates[-1])
  ax.set_ylabel('Number of people (millions)', color=INK); ax.set_xlabel('Date', color=INK)
  ax.set_title(f'SEIRV Model Simulation - {TITLES[name]}', color=INK, loc='left')
  ax.legend(frameon=False, loc='center left', fontsize=9)
  fig.tight_layout(); fig.savefig(f'{OUT}/new_seirv_{name}.png', dpi=150); plt.close(fig)

# (b) % with a first dose by tract on Apr 1 2021, (c) % infected during the rollout; one color scale per figure pair
blue = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
orange = plt.get_cmap('Oranges')
for key, cmap, title, fname in [('vaxMid', blue, 'Percentage of People with a First Dose by Tract (April 1, 2021)', 'new_vaccinated_apr2021'),
                                ('infRollout', orange, 'Percentage of People Infected During the Rollout by Tract (Dec 2020 - Nov 2021)', 'new_infected_rollout')]:
  vmin, vmax = np.percentile(np.concatenate([100 * res[k][key] for k in res]), [2, 98])
  for name in res:
    g = geo.copy(); g['v'] = 100 * res[name][key]
    fig, ax = plt.subplots(figsize=(8, 8))
    g.plot(column='v', cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, linewidth=0,
           legend=True, legend_kwds={'shrink': 0.6, 'label': '% of tract population'})
    ax.set_axis_off(); ax.set_title(f'{title}\n{TITLES[name]}', color=INK, fontsize=11, loc='left')
    fig.savefig(f'{OUT}/{fname}_{name}.png', dpi=150, bbox_inches='tight'); plt.close(fig)
