# Representation-of-a-Real-World-Problem-as-a-Markov-Decision-Process


## Aim

To represent an autonomous warehouse robot as a Markov Decision Process (MDP) by defining its states, actions, transition probabilities, rewards, and Python representation.

## Problem Statement

A coffee vending machine serves customers by accepting payment, preparing coffee, and dispensing it automatically.

### Problem Description

A coffee vending machine serves customers by accepting payment, preparing coffee, and dispensing it automatically. The machine must decide, at each stage, what action to take (wait for payment, prepare coffee, dispense coffee, or handle a fault) based on its current state, so as to maximize customer satisfaction (reward) while minimizing waste (unpaid dispensing, spoiled coffee, faults).

## MDP Components

A Markov Decision Process is represented as:

$$
MDP = (S, A, P, R, \gamma)
$$

Where:

| Symbol | Meaning |
|---|---|
| $S$ | Set of states |
| $A$ | Set of actions |
| $P$ | Transition probability function |
| $R$ | Reward function |
| $\gamma$ | Discount factor |

---

## State Space
```
S = {
    Idle,               # Machine is powered on and waiting for a customer
    Payment_Received,    # Customer has inserted payment; machine is ready to brew
    Brewing,             # Coffee is currently being prepared
    Dispensed,           # Coffee has been successfully dispensed to the customer
    Fault                # Machine has detected an error (e.g., no water/milk/beans, jam)
}
```


## Sample State

Sample State = Brewing

Meaning: The coffee vending machine is currently preparing coffee (grinding beans, heating and mixing water) after having received payment, and has not yet dispensed the cup to the customer.

## Action Space
```
A = {
    Wait,
    Accept_Payment,
    Prepare_Coffee,
    Dispense_Coffee,
    Reset_Fix_Fault
}
```

## Sample Action

Sample Action = Prepare_Coffee

Meaning: The machine initiates the coffee brewing process — this action is taken from the Payment_Received state, and leads probabilistically to either Brewing (success) or Fault (failure).

## Transition Probability
```
P(s' | s, a)

Idle --Wait--> Idle (0.7)

Idle --Wait--> Payment_Received (0.3)

Payment_Received --Prepare_Coffee--> Brewing (0.9)

Payment_Received --Prepare_Coffee--> Fault (0.1)

Brewing --Dispense_Coffee--> Dispensed (0.85)

Brewing --Dispense_Coffee--> Fault (0.15)

Dispensed --Wait--> Idle (1.0)

Fault --Reset_Fix_Fault--> Idle (0.8)

Fault --Reset_Fix_Fault--> Fault (0.2)
```
This means:

> Probability of reaching next state $s'$ after taking action $a$ in current state $s$.


## Reward Function
```
Idle --Wait--> Idle = 0

Idle --Wait--> Payment_Received = +2

Payment_Received --Prepare_Coffee--> Brewing = +5

Payment_Received --Prepare_Coffee--> Fault = -5

Brewing --Dispense_Coffee--> Dispensed = +15

Brewing --Dispense_Coffee--> Fault = -8

Dispensed --Wait--> Idle = +1

Fault --Reset_Fix_Fault--> Idle = +3

Fault --Reset_Fix_Fault--> Fault = -3
```

General form:

$$
R(s,a,s')
$$


## Graphical Representation

<img width="1180" height="456" alt="image" src="https://github.com/user-attachments/assets/701cf7bd-3592-4033-8ed6-b0b4c930351e" />

## Python Representation

```python
# MDP Representation using Python
print("Name: SOMESHWAR S   ")
print("Register Number: 212224040322 ")
states = ["Idle", "Payment_Received", "Brewing", "Dispensed", "Fault"]

actions = ["Wait", "Accept_Payment", "Prepare_Coffee", "Dispense_Coffee", "Reset_Fix_Fault"]

P = {
    "Idle": {
        "Wait": [
            ("Idle", 0.7, 0),
            ("Payment_Received", 0.3, 2)
        ]
    },
    "Payment_Received": {
        "Prepare_Coffee": [
            ("Brewing", 0.9, 5),
            ("Fault", 0.1, -5)
        ]
    },
    "Brewing": {
        "Dispense_Coffee": [
            ("Dispensed", 0.85, 15),
            ("Fault", 0.15, -8)
        ]
    },
    "Dispensed": {
        "Wait": [
            ("Idle", 1.0, 1)
        ]
    },
    "Fault": {
        "Reset_Fix_Fault": [
            ("Idle", 0.8, 3),
            ("Fault", 0.2, -3)
        ]
    }
}

gamma = 0.9

```
## Output
<img width="735" height="87" alt="image" src="https://github.com/user-attachments/assets/2057789a-2710-457a-b315-913a873f9606" />

<img width="735" height="842" alt="image" src="https://github.com/user-attachments/assets/82b3713d-7c69-4b30-b0b2-4db99b5448d1" />



## Result
The Coffee Vending Machine MDP was successfully implemented and analyzed. The defined states, actions, transition probabilities, and rewards accurately model the coffee dispensing process, including successful operation and fault recovery.


