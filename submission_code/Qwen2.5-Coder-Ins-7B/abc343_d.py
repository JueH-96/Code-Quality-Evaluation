# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])

events = []
for i in range(T):
    A = int(data[2 + 2 * i])
    B = int(data[3 + 2 * i])
    events.append((A, B))

# Initialize scores and a set to track unique scores
scores = [0] * N
unique_scores = set()

# Function to update the number of unique scores
def update_unique_scores():
    unique_scores.clear()
    for score in scores:
        unique_scores.add(score)
    return len(unique_scores)

# Process each event
results = []
for i in range(T):
    A, B = events[i]
    scores[A - 1] += B
    results.append(update_unique_scores())

# Print the results
for result in results:
    print(result)