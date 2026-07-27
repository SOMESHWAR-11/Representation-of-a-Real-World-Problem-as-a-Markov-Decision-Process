# Representation-of-a-Real-World-Problem-as-a-Markov-Decision-Process

## Aim

To identify a real-world sequential decision-making problem — traffic signal control at a road intersection — and represent it formally as a Markov Decision Process by defining its states, actions, rewards, transition probabilities, and Python representation.

## Problem Statement

### Problem Description

Urban intersections suffer from congestion because fixed-duration traffic lights cannot adapt to changing traffic volume. A four-way intersection (North, South, East, West) has vehicles arriving at each approach at random rates. A traffic signal controller must repeatedly decide whether to keep the current green phase running or switch to the other phase, aiming to minimize the total number of vehicles waiting at any point in time.

This is naturally sequential and stochastic: the controller's decision now affects queue lengths later, vehicle arrivals are random, and the goal is to minimize long-run congestion rather than a single-step outcome — exactly the setting an MDP is built for. An intelligent (RL-based) controller learns a policy that reacts to real-time queue conditions instead of running a rigid fixed-time cycle.

### MDP Components

A Markov Decision Process is represented as:

**MDP = (S, A, P, R, γ)**

| Symbol | Meaning |
|---|---|
| S | Set of states |
| A | Set of actions |
| P | Transition probability function |
| R | Reward function |
| γ | Discount factor |

## State Space

Each approach (North, South, East, West) has a queue length, discretized into three levels — **Low** (0–5 vehicles), **Medium** (6–15), **High** (16+) — combined with which phase currently has the green light.

```
S = {
    (N, S, E, W, phase) : N, S, E, W ∈ {Low, Medium, High},
                           phase ∈ {NS_Green, EW_Green}
}
```

This gives 3⁴ × 2 = **162 states** in total.

### Sample State

```
s = (High, Low, Medium, Low, NS_Green)
```
North has a high queue, South is low, East is medium, West is low, and North–South currently has the green light.

## Action Space

```
A = {
    Keep_Current_Phase,
    Switch_Phase
}
```

### Sample Action

```
a = Switch_Phase
```

## Transition Probability

**P(s'|s,a)** — the probability of reaching state s' after taking action a in state s.

When a direction has the green light, its queue tends to shrink (vehicles are discharged); when it's red, its queue tends to grow (new arrivals accumulate, no departures). This is modeled per direction as a two-outcome probability distribution — 70% chance a green-lit queue drops one level (30% it stays), and 60% chance a red-lit queue rises one level (40% it stays) — then combined across all four directions assuming independence. Switching phase flips which pair (N–S or E–W) is treated as green for the next step.

For the sample state and action above, this yields 4 reachable next-states, with the most likely being **s' = (High, Medium, Low, Low, EW_Green)** at **P ≈ 0.050**.

## Reward Function

**R(s,a,s')** = −(total vehicles queued in s') − 5 if the action was Switch_Phase

Queue levels are mapped to representative vehicle counts (Low=3, Medium=10, High=20) to compute a numeric penalty. The agent is penalized in proportion to total congestion, plus a small extra penalty for switching phases — modeling the real-world lost time (yellow/all-red clearance interval) every switch costs.

For **s=(High,Low,Medium,Low,NS_Green)**, **a=Switch_Phase**, most likely **s'=(High,Medium,Low,Low,EW_Green)**: **R = −41**.

## Discount Factor

γ = 0.9 — traffic control is a continuing (non-episodic) process, so future congestion still matters but is discounted since farther-future traffic is less certain.

## Graphical Representation

The diagram above shows the sample state, the two available actions, and — for each action — the most likely resulting state along with its transition probability P(s'|s,a) and reward R(s,a,s').

## Python Representation

```python
import itertools

print("Name: Someshwar")
print("Register Number: <Enter Register Number>")
print()

# ---------------- STATE SPACE ----------------
QUEUE_LEVELS = ["Low", "Medium", "High"]   # Low:0-5, Medium:6-15, High:16+ vehicles
PHASES = ["NS_Green", "EW_Green"]

def generate_states():
    states = []
    for n, s, e, w in itertools.product(QUEUE_LEVELS, repeat=4):
        for phase in PHASES:
            states.append((n, s, e, w, phase))
    return states

S = generate_states()

# ---------------- ACTION SPACE ----------------
A = ["Keep_Current_Phase", "Switch_Phase"]

# Numeric mapping used only for computing rewards
QUEUE_VALUE = {"Low": 3, "Medium": 10, "High": 20}
SWITCH_PENALTY = 5
GAMMA = 0.9

# ---------------- REWARD FUNCTION ----------------
def R(state, action, next_state):
    n, s, e, w, phase = next_state
    total_wait = QUEUE_VALUE[n] + QUEUE_VALUE[s] + QUEUE_VALUE[e] + QUEUE_VALUE[w]
    reward = -total_wait
    if action == "Switch_Phase":
        reward -= SWITCH_PENALTY
    return reward

# ---------------- TRANSITION PROBABILITY ----------------
def transition_probabilities(state, action):
    n, s, e, w, phase = state
    if action == "Switch_Phase":
        new_phase = "EW_Green" if phase == "NS_Green" else "NS_Green"
    else:
        new_phase = phase

    def next_level_dist(level, is_green):
        idx = QUEUE_LEVELS.index(level)
        if is_green:
            drop = max(idx - 1, 0)
            return {QUEUE_LEVELS[drop]: 0.7, QUEUE_LEVELS[idx]: 0.3}
        else:
            rise = min(idx + 1, 2)
            return {QUEUE_LEVELS[rise]: 0.6, QUEUE_LEVELS[idx]: 0.4}

    ns_green = new_phase == "NS_Green"
    dist_n = next_level_dist(n, ns_green)
    dist_s = next_level_dist(s, ns_green)
    dist_e = next_level_dist(e, not ns_green)
    dist_w = next_level_dist(w, not ns_green)

    transitions = {}
    for nn, pn in dist_n.items():
        for ns_, ps in dist_s.items():
            for ne, pe in dist_e.items():
                for nw, pw in dist_w.items():
                    prob = pn * ps * pe * pw
                    next_state = (nn, ns_, ne, nw, new_phase)
                    transitions[next_state] = round(prob, 5)
    return transitions

# ---------------- DEMO / OUTPUT ----------------
print(f"Total number of states  |S| = {len(S)}")
print(f"Total number of actions |A| = {len(A)}")
print(f"Discount factor gamma = {GAMMA}")
print()
print("First 5 states in S:")
for st in S[:5]:
    print("  ", st)
print()

sample_state = ("High", "Low", "Medium", "Low", "NS_Green")
sample_action = "Switch_Phase"

print("Sample state  s =", sample_state)
print("Sample action a =", sample_action)
print()

trans = transition_probabilities(sample_state, sample_action)
print(f"Number of reachable next-states from (s,a): {len(trans)}")
print("Top 5 most probable transitions P(s'|s,a):")
for ns_, p in sorted(trans.items(), key=lambda x: -x[1])[:5]:
    print(f"   s' = {ns_}  ->  P = {p}")
print()

most_likely_next = max(trans.items(), key=lambda x: x[1])[0]
reward = R(sample_state, sample_action, most_likely_next)
print("Most likely next state s' =", most_likely_next)
print(f"R(s,a,s') = {reward}")
```

## Output

```
Name: Someshwar
Register Number: <212224040322>

Total number of states  |S| = 162
Total number of actions |A| = 2
Discount factor gamma = 0.9

First 5 states in S:
    ('Low', 'Low', 'Low', 'Low', 'NS_Green')
    ('Low', 'Low', 'Low', 'Low', 'EW_Green')
    ('Low', 'Low', 'Low', 'Medium', 'NS_Green')
    ('Low', 'Low', 'Low', 'Medium', 'EW_Green')
    ('Low', 'Low', 'Low', 'High', 'NS_Green')

Sample state  s = ('High', 'Low', 'Medium', 'Low', 'NS_Green')
Sample action a = Switch_Phase

Number of reachable next-states from (s,a): 4
Top 5 most probable transitions P(s'|s,a):
   s' = ('High', 'Medium', 'Low', 'Low', 'EW_Green')  ->  P = 0.0504
   s' = ('High', 'Low', 'Low', 'Low', 'EW_Green')  ->  P = 0.0336
   s' = ('High', 'Medium', 'Medium', 'Low', 'EW_Green')  ->  P = 0.0216
   s' = ('High', 'Low', 'Medium', 'Low', 'EW_Green')  ->  P = 0.0144

Most likely next state s' = ('High', 'Medium', 'Low', 'Low', 'EW_Green')
R(s,a,s') = -41
```

## Result

The intelligent traffic signal control problem was successfully modeled as a Markov Decision Process with 162 states (combining discretized queue levels for four approaches with two signal phases), 2 actions (Keep_Current_Phase, Switch_Phase), a probabilistic transition function based on arrival/departure dynamics, and a congestion-minimizing reward function. The Python implementation confirmed the state/action counts and correctly computed transition probabilities and rewards for a sample state-action pair, verifying the MDP formulation.

---
