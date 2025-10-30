import sys
import collections

# Read all input data
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1

# Store dice information: each die has K and a counter of face frequencies
dice = []
for _ in range(N):
    K = int(data[index])
    index += 1
    faces = list(map(int, data[index:index + K]))
    index += K
    counter = collections.Counter(faces)
    dice.append((K, counter))

# Initialize maximum probability
max_prob = 0.0

# Iterate over all pairs of dice
for i in range(N):
    for j in range(i + 1, N):
        K_i, counter_i = dice[i]
        K_j, counter_j = dice[j]
        
        # Compute the dot sum of frequencies, iterating over the smaller counter for efficiency
        len_i = len(counter_i)
        len_j = len(counter_j)
        if len_i <= len_j:
            dot_sum = sum(counter_i[x] * counter_j[x] for x in counter_i)
        else:
            dot_sum = sum(counter_j[x] * counter_i[x] for x in counter_j)
        
        # Calculate probability and update maximum if necessary
        prob = dot_sum / (K_i * K_j)
        if prob > max_prob:
            max_prob = prob

# Output the maximum probability with high precision
print("{:.15f}".format(max_prob))