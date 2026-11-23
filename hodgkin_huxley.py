"""Hodgkin-Huxley neuron: action potential, firing-rate curve, refractory period, channel blockers.

Model: Hodgkin and Huxley (1952), squid giant axon, modern voltage convention (rest = -65 mV).
Units: mV, ms, uA/cm^2, mS/cm^2, uF/cm^2. Integration: 4th-order Runge-Kutta.

Run:  python hodgkin_huxley.py            (takes about a minute)
Needs: pip install numpy matplotlib
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C, GNA, GK, GL = 1.0, 120.0, 36.0, 0.3
ENA, EK, EL = 50.0, -77.0, -54.387


def vtrap(x, y):
    """x / (1 - exp(-x/y)), with the removable singularity at x = 0 handled."""
    if abs(x / y) < 1e-6:
        return y * (1.0 + x / (2.0 * y))
    return x / (1.0 - math.exp(-x / y))


def rates(v):
    an = 0.01 * vtrap(v + 55.0, 10.0)
    bn = 0.125 * math.exp(-(v + 65.0) / 80.0)
    am = 0.1 * vtrap(v + 40.0, 10.0)
    bm = 4.0 * math.exp(-(v + 65.0) / 18.0)
    ah = 0.07 * math.exp(-(v + 65.0) / 20.0)
    bh = 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))
    return an, bn, am, bm, ah, bh


def rest_state(v=-65.0):
    an, bn, am, bm, ah, bh = rates(v)
    return v, am / (am + bm), ah / (ah + bh), an / (an + bn)


def deriv(s, i_ext, gna, gk):
    v, m, h, n = s
    an, bn, am, bm, ah, bh = rates(v)
    dv = (i_ext - gna * m ** 3 * h * (v - ENA) - gk * n ** 4 * (v - EK) - GL * (v - EL)) / C
    return (dv, am * (1 - m) - bm * m, ah * (1 - h) - bh * h, an * (1 - n) - bn * n)


def simulate(stim, t_end, dt=0.01, gna_scale=1.0, gk_scale=1.0):
    """stim(t) -> injected current. Returns time, V, m, h, n arrays."""
    gna, gk = GNA * gna_scale, GK * gk_scale
    steps = int(round(t_end / dt))
    out = np.empty((steps + 1, 4))
    s = rest_state()
    out[0] = s
    for k in range(steps):
        t = k * dt
        k1 = deriv(s, stim(t), gna, gk)
        s2 = tuple(a + 0.5 * dt * b for a, b in zip(s, k1))
        k2 = deriv(s2, stim(t + 0.5 * dt), gna, gk)
        s3 = tuple(a + 0.5 * dt * b for a, b in zip(s, k2))
        k3 = deriv(s3, stim(t + 0.5 * dt), gna, gk)
        s4 = tuple(a + dt * b for a, b in zip(s, k3))
        k4 = deriv(s4, stim(t + dt), gna, gk)
        s = tuple(a + dt / 6.0 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(s, k1, k2, k3, k4))
        out[k + 1] = s
    return np.arange(steps + 1) * dt, out[:, 0], out[:, 1], out[:, 2], out[:, 3]


def pulse(start, dur, amp):
    return lambda t: amp if start <= t < start + dur else 0.0


def upward_crossings(t, v, level=0.0):
    idx = np.where((v[:-1] < level) & (v[1:] >= level))[0]
    return t[idx]


def spike_width(t, v, v_rest=-65.0):
    """Width of the first spike at half of its amplitude, in ms (nan if no spike)."""
    if v.max() < 0:
        return float("nan")
    half = v_rest + 0.5 * (v.max() - v_rest)
    up = np.where((v[:-1] < half) & (v[1:] >= half))[0]
    if len(up) == 0:
        return float("nan")
    down = np.where((v[:-1] >= half) & (v[1:] < half) & (np.arange(len(v) - 1) > up[0]))[0]
    return float(t[down[0]] - t[up[0]]) if len(down) else float("nan")


def threshold_amplitude(make_stim, t_end, lo=0.0, hi=500.0, needed=1, iters=12, dt=0.02, **kw):
    """Smallest pulse amplitude that produces `needed` spikes (bisection); inf if even `hi` fails."""
    def fires(a):
        t, v, *_ = simulate(make_stim(a), t_end, dt, **kw)
        return len(upward_crossings(t, v)) >= needed
    if not fires(hi):
        return float("inf")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if fires(mid) else (mid, hi)
    return hi


def exp1_action_potential(out, log):
    t, v, m, h, n = simulate(pulse(5.0, 1.0, 10.0), 30.0)
    gna, gk = GNA * m ** 3 * h, GK * n ** 4
    thr = threshold_amplitude(lambda a: pulse(5.0, 1.0, a), 30.0)
    fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax[0].plot(t, v, color="k"); ax[0].set_ylabel("Membrane potential (mV)")
    ax[1].plot(t, m, label="m (Na activation)"); ax[1].plot(t, h, label="h (Na inactivation)")
    ax[1].plot(t, n, label="n (K activation)"); ax[1].set_ylabel("Gating variable"); ax[1].legend(fontsize=8)
    ax[2].plot(t, gna, label="g_Na"); ax[2].plot(t, gk, label="g_K")
    ax[2].set_ylabel("Conductance (mS/cm$^2$)"); ax[2].set_xlabel("Time (ms)"); ax[2].legend(fontsize=8)
    fig.suptitle("Action potential after a 1 ms, 10 uA/cm$^2$ pulse"); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_action_potential.png"), dpi=200); plt.close(fig)
    log.append(f"[1] Action potential: peak {v.max():.1f} mV at {t[v.argmax()]:.2f} ms; "
               f"minimum after spike {v[t > 6].min():.1f} mV; width at half amplitude {spike_width(t, v):.2f} ms; "
               f"threshold for a 1 ms pulse about {thr:.2f} uA/cm^2; peak g_Na {gna.max():.1f}, peak g_K {gk.max():.1f} mS/cm^2")


def exp2_fi_curve(out, log):
    currents = np.arange(0.0, 20.1, 1.0)
    rates_hz = []
    for i_ext in currents:
        t, v, *_ = simulate(lambda _t, a=i_ext: a, 500.0, dt=0.02)
        ups = upward_crossings(t, v)
        rates_hz.append(np.sum(ups >= 100.0) / 0.4)
    firing = [c for c, r in zip(currents, rates_hz) if r > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(currents, rates_hz, "o-", color="k")
    ax.set_xlabel("Constant injected current (uA/cm$^2$)"); ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Firing rate versus current"); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_fI_curve.png"), dpi=200); plt.close(fig)
    first = f"{firing[0]:.0f} uA/cm^2" if firing else "none up to 20 uA/cm^2"
    log.append(f"[2] f-I curve: lowest current with sustained firing (step 1 uA/cm^2): {first}; "
               + "; ".join(f"{c:.0f}: {r:.0f} Hz" for c, r in zip(currents, rates_hz) if c in (5, 8, 10, 15, 20)))


def exp3_refractory(out, log):
    intervals = np.arange(2.0, 26.0, 1.0)
    thr = []
    for gap in intervals:
        make = lambda a, gap=gap: (lambda t: 10.0 if 5.0 <= t < 6.0 else (a if 5.0 + gap <= t < 6.0 + gap else 0.0))
        thr.append(threshold_amplitude(make, 5.0 + gap + 15.0, needed=2, iters=10))
    thr = np.array(thr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ok = np.isfinite(thr)
    ax.plot(intervals[ok], thr[ok], "o-", color="k")
    ax.set_yscale("log"); ax.set_xlabel("Interval between the two pulses (ms)")
    ax.set_ylabel("Threshold amplitude of second 1 ms pulse (uA/cm$^2$)")
    ax.set_title("Refractory period (first pulse: 10 uA/cm$^2$)"); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_refractory.png"), dpi=200); plt.close(fig)
    first_ok = intervals[ok][0] if ok.any() else float("nan")
    base = thr[ok][-1] if ok.any() else float("nan")
    log.append(f"[3] Refractory period: no second spike for intervals up to {intervals[~ok].max() if (~ok).any() else 0:.0f} ms "
               f"even with 500 uA/cm^2; shortest interval with a second spike: {first_ok:.0f} ms; "
               f"threshold at {intervals[ok][-1]:.0f} ms: {base:.2f} uA/cm^2; "
               + "; ".join(f"{g:.0f} ms: {x:.1f}" for g, x in zip(intervals, thr) if g in (3, 5, 8, 10, 15, 25) and np.isfinite(x)))


def exp4_blockers(out, log):
    stim = pulse(5.0, 1.0, 20.0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    lines = []
    for sc in (1.0, 0.5, 0.3, 0.1):
        t, v, *_ = simulate(stim, 30.0, gna_scale=sc)
        ax[0].plot(t, v, label=f"g_Na x {sc:g}")
        lines.append(f"g_Na x {sc:g}: peak {v.max():.1f} mV")
    for sc in (1.0, 0.5, 0.1):
        t, v, *_ = simulate(stim, 30.0, gk_scale=sc)
        ax[1].plot(t, v, label=f"g_K x {sc:g}")
        lines.append(f"g_K x {sc:g}: peak {v.max():.1f} mV, width {spike_width(t, v):.2f} ms")
    ax[0].set_title("Sodium channel block (TTX-like)"); ax[1].set_title("Potassium channel block (TEA-like)")
    for a in ax:
        a.set_xlabel("Time (ms)"); a.legend(fontsize=8)
    ax[0].set_ylabel("Membrane potential (mV)"); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_channel_blockers.png"), dpi=200); plt.close(fig)
    log.append("[4] Channel blockers (20 uA/cm^2, 1 ms pulse): " + "; ".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = []
    for f in (exp1_action_potential, exp2_fi_curve, exp3_refractory, exp4_blockers):
        print("running", f.__name__, flush=True)
        f(a.out, log)
    text = "\n".join(log)
    print(text)
    open(os.path.join(a.out, "results.txt"), "w").write(text + "\n")
