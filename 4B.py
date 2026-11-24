from collections import defaultdict, Counter
import numpy as np

weather = [
    "Sunny", "Cloudy", "Rainy", "Rainy",
    "Cloudy", "Sunny", "Cloudy", "Rainy",
    "Sunny", "Sunny", "Cloudy", "Rainy",
    "Rainy", "Cloudy", "Sunny", "Cloudy"
]

counts = defaultdict(Counter)
probs = {}

# Count transitions
for i in range(2,len(weather)):
    pair = (weather[i-2], weather[i-1])
    counts[pair][weather[i]] += 1

# Calculate probabilities
for pair,c in counts.items():
    total = sum(c.values())
    probs[pair] = {}

    for state in c:
        probs[pair][state] = c[state]/total

# Display probabilities
for pair,p in probs.items():
    print(pair, "->")

    for state in p:
        print(state, ":", round(p[state]*100, 2), "%")

# Predict next state
pair = (weather[-2],weather[-1])

if pair in probs:
    states = list(probs[pair].keys())
    p = list(probs[pair].values())

    prediction = np.random.choice(states,p=p)

    print("Previous states:", pair)
    print("Predicted state:", prediction)
else:
    print("Insufficient data")

