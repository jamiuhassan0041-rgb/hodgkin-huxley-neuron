# Hodgkin-Huxley neuron simulation

Independent learning project by Jamiu Hassan, prepared as part of an application to a neuroscience master's programme. No affiliation with any programme is implied.

## Question
How do voltage-gated sodium and potassium channels produce an action potential, and how do the model's behaviours (threshold, repetitive firing, refractoriness, channel block) follow from them?

## Model
The Hodgkin and Huxley (1952) squid giant axon model, written in Python (NumPy) from the standard equations, with 4th-order Runge-Kutta integration (dt = 0.01 ms for single spikes, 0.02 ms for the longer experiments). Resting potential -65 mV, no temperature scaling. Parameters: C = 1 uF/cm2, g_Na = 120, g_K = 36, g_L = 0.3 mS/cm2, E_Na = 50, E_K = -77, E_L = -54.387 mV.

## Experiments and results

### 1. Action potential
A 1 ms, 10 uA/cm2 pulse produced a spike peaking at +39.1 mV, with an undershoot to -76.2 mV and a width of 1.45 ms at half amplitude. The weakest 1 ms pulse that triggered a spike was about 6.96 uA/cm2. Peak g_Na was 31.4 and peak g_K 12.5 mS/cm2.

![Action potential](results/fig1_action_potential.png)

The pulse depolarises the membrane past threshold. Sodium activation (m) is fast, so g_Na rises first, and the sodium influx depolarises the cell further, which opens more sodium channels (positive feedback). The voltage rushes toward E_Na (+50 mV) but peaks at +39 mV because sodium inactivation (h) is already falling and potassium activation (n) is rising. g_K peaks after g_Na because n is slower than m. Outward potassium current then repolarises the membrane, and because n decays slowly the voltage undershoots to -76 mV, close to E_K (-77 mV), before recovering. Peak g_Na is only about a quarter of its maximum (120 mS/cm2) because m and h are never both large at the same moment.

### 2. Firing rate versus current
With constant current, the neuron was silent at 6 uA/cm2 and below, and fired at 60 Hz at 7 uA/cm2 (the first current tested that produced firing). The rate then rose slowly: 70 Hz at 10, 80 Hz at 15 and 88 Hz at 20 uA/cm2.

![f-I curve](results/fig2_fI_curve.png)

The rate jumps from zero to about 60 Hz and never fires slowly. This is the abrupt onset known as class 2 (type II) excitability. A likely reason is that the time between spikes is set mainly by how fast h and n recover after each spike, not by how close the current is to threshold, so the neuron cannot fire more slowly than that recovery allows. I did not test this explanation directly. Small up-and-down steps in the curve (for example around 16-17 uA/cm2) are counting noise: rates are counted over 0.4 s, so one spike is 2.5 Hz.

### 3. Refractory period
A first 1 ms pulse of 10 uA/cm2 was followed by a second 1 ms pulse after a variable interval, and I searched for the weakest second pulse that produced a second spike. For intervals up to 5 ms no second spike occurred even with 500 uA/cm2 (the absolute refractory period in this model). A second spike first appeared at 6 ms, needing over 200 uA/cm2. The threshold then fell as the interval grew: 62.5 uA/cm2 at 8 ms, 30.8 at 10 ms, 9.3 at 15 ms and about 7 at 25 ms, close to the resting threshold of 6.96 (relative refractory period).

![Refractory period](results/fig3_refractory.png)

Both phases follow from the gating variables after a spike. Soon after the spike most sodium channels are inactivated (h is low), so even a very strong pulse cannot regenerate a spike. As h recovers, a spike becomes possible, but potassium channels are still open (n is high), so extra current is needed to overcome the outward potassium current. As h and n return to rest, the threshold returns to its normal value.

### 4. Channel blockers
With a 20 uA/cm2, 1 ms pulse, scaling g_Na down (TTX-like) lowered the peak from 40.5 mV (normal) to 27.8 mV at 50%, -2.2 mV at 30% (no real spike) and -51.3 mV at 10% (a passive depolarisation only). Scaling g_K down (TEA-like) made the spike taller and wider: width 1.48 ms normally, 1.82 ms at 50% and 3.21 ms at 10%.

![Channel blockers](results/fig4_channel_blockers.png)

Without enough sodium conductance the positive feedback that drives the upstroke fails, so the spike shrinks and then disappears. With less potassium conductance there is less outward current to repolarise the membrane, so the spike lasts longer.

## Limitations
- Single compartment, so there is no propagation along an axon.
- Squid axon parameters without temperature scaling, so timings (for example the 5 ms absolute refractory period) do not match mammalian neurons at body temperature.
- Channel block is modelled only as a reduced maximal conductance.
- Thresholds in experiment 3 were found by bisection with a resolution of about 0.5 uA/cm2, so small differences at long intervals are not meaningful.
- Results were not compared against experimental recordings.

## How to run
```
pip install numpy matplotlib
python hodgkin_huxley.py
```
The simulation is deterministic, so it reproduces the numbers above. It takes about a minute and writes four figures and `results.txt` to `results/`.

## Credits
Hodgkin A. L. and Huxley A. F. (1952), J. Physiol. 117:500-544. Voltage convention and rate functions as in standard textbook presentations (for example Dayan and Abbott, Theoretical Neuroscience). The code and this write-up were drafted with the help of an AI assistant (Claude); I ran the simulations and checked the results against the figures.
