import sys, pickle
exec(open(sys.argv[1]).read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
tag = f'rho{RHO_LATE}' + ('_cap' if UPTAKE_CAP else '') + ('_prevax' if PREVAX_FACTORS else '')
saved = pickle.load(open(f'{SCR}/{FIT_FILE}', 'rb')); zf = saved['zf']
rr = tractFactors(zf)
def run(alloc, until=END):
  y = initial(saved['seed'], rr); ons = []
  for ws, we, b in saved['windows']:
    if ws >= until: break
    y, on = step_days(y, ws, min(we, until), b, rr, alloc); ons.append(on)
  return np.concatenate(ons), y
res = dict(onsets=saved['onsets'])
VSTART, MID = 286, 397   # rollout starts Dec 11 2020; mid-rollout snapshot Apr 1 2021
q = pd.qcut(inc, 4, labels=False); QN = ['Poorest', 'Q2', 'Q3', 'Richest']
dates = DAY0 + pd.to_timedelta(np.arange(END), 'D')

# ---------- fit quality ----------
rep = res['onsets'] * np.array([rho(dd, rhoEarly) for dd in range(END)])[:, None]
fig, axes = plt.subplots(3, 1, figsize=(12, 13))
axes[0].plot(dates, citySmooth, color='k', label='NYC reported cases (7-day avg)')
axes[0].plot(dates, rep.sum(1), color='C3', label='Model reported cases')
axes[0].set_ylabel('Cases per day'); axes[0].legend(); axes[0].grid(True); axes[0].set_title('Model fit to NYC reported cases')
for k, b in enumerate(BOROS):
  axes[1].plot(dates, pd.Series(boroCases[b]).rolling(7, center=True, min_periods=1).mean() / pop[boro == b].sum() * 1e5, color=f'C{k}', lw=1)
  axes[1].plot(dates, rep[:, boro == b].sum(1) / pop[boro == b].sum() * 1e5, color=f'C{k}', ls='--', label=b)
axes[1].set_ylabel('Cases per day per 100k'); axes[1].set_title('By borough: data (solid) vs model (dashed)'); axes[1].legend(); axes[1].grid(True)
mr = modelRel(res['onsets'])
for p, name in enumerate(['Mar-May 2020', 'Jun-Nov 2020', 'Dec 2020-May 2021', 'Jun-Aug 2021', 'Sep-Nov 2021']):
  axes[2].scatter(obsRel[p], mr[p], s=8, label=name, alpha=0.7)
axes[2].plot([0, 3], [0, 3], color='k', lw=0.8); axes[2].set_xlim(0, 3); axes[2].set_ylim(0, 3)
axes[2].set_xlabel('Observed ZIP case rate / city rate'); axes[2].set_ylabel('Model ZIP case rate / city rate')
axes[2].set_title('ZIP-level fit by period'); axes[2].legend(); axes[2].grid(True)
fig.tight_layout(); fig.savefig(f'{SCR}/fit2_{tag}.png', dpi=100)

# ---------- scenarios ----------
scen = {}
for name, alloc in [('No vaccine', None), ('Actual rollout', actualAllocation), ('Uniform', uniformAllocation), ('Income-based', incomeAllocation),
                    ('Susceptible-first', susceptibleAllocation), ('Commuting-based', commuteAllocation), ('Hotspot', hotspotAllocation(rr))]:
  onsets, _ = run(alloc)
  _, y = run(alloc, MID)
  scen[name] = dict(onsets=onsets, vaxMid=y[1] + y[3] + y[5] + y[7])
pickle.dump(scen, open(f'{SCR}/scenarios2_{tag}.pkl', 'wb'))

print(f'reporting rate after Jun 2020 = {RHO_LATE}; ever infected by Nov 2021 (actual rollout) = {scen["Actual rollout"]["onsets"].sum()/pop.sum():.1%}')
print('Infections Dec 11 2020 - Nov 30 2021:')
for name, s in scen.items():
  inf = s['onsets'][VSTART:].sum(0)
  print(f'  {name:17s} {inf.sum():9.0f} | by income quartile ' + ' '.join(f'{QN[k]} {100*inf[q==k].sum()/pop[q==k].sum():5.1f}%' for k in range(4))
        + ' | ' + ' '.join(f'{BOROS[b]} {100*inf[boro==b].sum()/pop[boro==b].sum():5.1f}%' for b in BOROS))
print('First-dose coverage on Apr 1 2021:')
for name, s in scen.items():
  if name == 'No vaccine': continue
  print(f'  {name:17s} ' + ' '.join(f'{QN[k]} {100*s["vaxMid"][q==k].sum()/pop[q==k].sum():5.1f}%' for k in range(4)))
print('Fitted ZIP transmission factor vs income (pop-weighted mean by income quartile, per period):')
for p in range(len(PERIODS) - 1):
  print(f'  period {p}: ' + ' '.join(f'{QN[k]} {np.average(rr[p][q==k], weights=pop[q==k]):.2f}' for k in range(4)))

fig, ax = plt.subplots(figsize=(12, 5))
for name, col, ls in [('No vaccine', 'gray', '--'), ('Actual rollout', 'k', '-'), ('Uniform', 'C0', '-'), ('Income-based', 'C1', '-'),
                      ('Susceptible-first', 'C2', '-'), ('Commuting-based', 'C3', '-'), ('Hotspot', 'C4', '-')]:
  ax.plot(dates[VSTART - 30:], scen[name]['onsets'][VSTART - 30:].sum(1), color=col, ls=ls, label=name)
ax.set_ylabel('New infections per day (true, not reported)'); ax.legend(); ax.grid(True)
ax.set_title('NYC infections during the vaccine rollout, by allocation strategy')
fig.tight_layout(); fig.savefig(f'{SCR}/scenarios2_time_{tag}.png', dpi=100)

fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(4); wdt = 0.12
for k, (name, col) in enumerate([('No vaccine', 'gray'), ('Actual rollout', 'k'), ('Uniform', 'C0'), ('Income-based', 'C1'),
                                 ('Susceptible-first', 'C2'), ('Commuting-based', 'C3'), ('Hotspot', 'C4')]):
  inf = scen[name]['onsets'][VSTART:].sum(0)
  ax.bar(x + (k - 3) * wdt, [100 * inf[q == j].sum() / pop[q == j].sum() for j in range(4)], wdt, color=col, label=name)
ax.set_xticks(x, QN); ax.set_xlabel('Income quartile (tracts)')
ax.set_ylabel('% of population infected, Dec 2020-Nov 2021'); ax.legend(); ax.grid(axis='y')
ax.set_title('Infections during the rollout by income quartile')
fig.tight_layout(); fig.savefig(f'{SCR}/scenarios2_quartiles_{tag}.png', dpi=100)
