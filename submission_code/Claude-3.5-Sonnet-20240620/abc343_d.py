# YOUR CODE HERE
from collections import defaultdict

# Read input
N, T = map(int, input().split())
events = [tuple(map(int, input().split())) for _ in range(T)]

# Initialize scores and unique score count
scores = defaultdict(int)
unique_scores = set([0])

# Process events and count unique scores
for i, (player, points) in enumerate(events, 1):
    scores[player] += points
    unique_scores.add(scores[player])
    print(len(unique_scores))