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

# ============================================================
# MDP = (S, A, P, R, gamma)
# Traffic Signal Control at a Four-Way Intersection
# ============================================================

print("Name: Someshwar S")
print("Register Number: 212224040322")
print()

# ---------------- STATE SPACE ----------------

QUEUE_LEVELS = ["Low", "Medium", "High"]
PHASES = ["NS_Green", "EW_Green"]

# Low = 0-5 vehicles
# Medium = 6-15 vehicles
# High = 16+ vehicles

def generate_states():
    states = []

    for n, s, e, w in itertools.product(QUEUE_LEVELS, repeat=4):
        for phase in PHASES:
            states.append((n, s, e, w, phase))

    return states


S = generate_states()

# Give every state a unique integer ID
STATE_ID = {state: i for i, state in enumerate(S)}
ID_STATE = {i: state for i, state in enumerate(S)}

# ---------------- ACTION SPACE ----------------

A = [
    "Keep_Current_Phase",
    "Switch_Phase"
]

ACTION_ID = {
    "Keep_Current_Phase": 0,
    "Switch_Phase": 1
}

# ---------------- REWARD PARAMETERS ----------------

QUEUE_VALUE = {
    "Low": 3,
    "Medium": 10,
    "High": 20
}

SWITCH_PENALTY = 5
GAMMA = 0.9


# ============================================================
# REWARD FUNCTION
# R(s,a,s') = -(total vehicles queued in s') - 5
#             if action is Switch_Phase
# ============================================================

def reward_function(state, action, next_state):

    n, s, e, w, phase = next_state

    total_wait = (
        QUEUE_VALUE[n]
        + QUEUE_VALUE[s]
        + QUEUE_VALUE[e]
        + QUEUE_VALUE[w]
    )

    reward = -total_wait

    if action == "Switch_Phase":
        reward -= SWITCH_PENALTY

    return reward


# ============================================================
# TRANSITION DISTRIBUTION
# ============================================================

def next_level_distribution(level, is_green):

    idx = QUEUE_LEVELS.index(level)

    if is_green:
        # 70% chance queue decreases one level
        # 30% chance queue remains unchanged
        if idx == 0:
            return {
                "Low": 1.0
            }

        return {
            QUEUE_LEVELS[idx - 1]: 0.7,
            QUEUE_LEVELS[idx]: 0.3
        }

    else:
        # 60% chance queue increases one level
        # 40% chance queue remains unchanged
        if idx == 2:
            return {
                "High": 1.0
            }

        return {
            QUEUE_LEVELS[idx + 1]: 0.6,
            QUEUE_LEVELS[idx]: 0.4
        }


# ============================================================
# BUILD P IN THE REQUIRED FORMAT
#
# P[state][action] =
# [
#     (probability, next_state, reward, done),
#     ...
# ]
# ============================================================

P = {}

for state in S:

    P[STATE_ID[state]] = {}

    for action in A:

        n, s, e, w, phase = state

        # Determine the signal phase for next step
        if action == "Switch_Phase":
            if phase == "NS_Green":
                new_phase = "EW_Green"
            else:
                new_phase = "NS_Green"
        else:
            new_phase = phase

        # Determine which directions have green
        ns_green = (new_phase == "NS_Green")

        dist_n = next_level_distribution(n, ns_green)
        dist_s = next_level_distribution(s, ns_green)

        dist_e = next_level_distribution(e, not ns_green)
        dist_w = next_level_distribution(w, not ns_green)

        transitions = []

        # Combine independent probabilities
        for nn, pn in dist_n.items():
            for ss, ps in dist_s.items():
                for ee, pe in dist_e.items():
                    for ww, pw in dist_w.items():

                        probability = pn * ps * pe * pw

                        next_state = (
                            nn,
                            ss,
                            ee,
                            ww,
                            new_phase
                        )

                        next_state_id = STATE_ID[next_state]

                        reward = reward_function(
                            state,
                            action,
                            next_state
                        )

                        # Traffic control is a continuing process,
                        # therefore no terminal states.
                        done = False

                        transitions.append(
                            (
                                round(probability, 5),
                                next_state_id,
                                reward,
                                done
                            )
                        )

        P[STATE_ID[state]][ACTION_ID[action]] = transitions


# ============================================================
# OUTPUT
# ============================================================

print("MDP Representation")
print("------------------")

print("Number of states |S| =", len(S))
print("Number of actions |A| =", len(A))
print("Discount factor gamma =", GAMMA)

print()

# Sample state
sample_state = (
    "High",
    "Low",
    "Medium",
    "Low",
    "NS_Green"
)

sample_state_id = STATE_ID[sample_state]

print("Sample state:")
print("State ID =", sample_state_id)
print("State =", sample_state)

print()

# Sample action
sample_action = "Switch_Phase"
sample_action_id = ACTION_ID[sample_action]

print("Sample action:")
print("Action ID =", sample_action_id)
print("Action =", sample_action)

print()

# Display P for the sample state/action
print("P[sample_state][sample_action] =")

for transition in P[sample_state_id][sample_action_id]:

    probability, next_state_id, reward, done = transition

    print(
        " ",
        (
            probability,
            next_state_id,
            reward,
            done
        ),
        "->",
        ID_STATE[next_state_id]
    )
```

## Output

<img width="920" height="484" alt="image" src="https://github.com/user-attachments/assets/9c4790ee-23e9-4a12-ae93-dbb1721f28b0" />


## Result

The intelligent traffic signal control problem was successfully modeled as a Markov Decision Process with 162 states (combining discretized queue levels for four approaches with two signal phases), 2 actions (Keep_Current_Phase, Switch_Phase), a probabilistic transition function based on arrival/departure dynamics, and a congestion-minimizing reward function. The Python implementation confirmed the state/action counts and correctly computed transition probabilities and rewards for a sample state-action pair, verifying the MDP formulation.

---
