# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
events = [(int(data[2 * i + 2]) - 1, int(data[2 * i + 3])) for i in range(T)]

scores = [0] * N
score_set = set()
result = []

for i, (player, points) in enumerate(events):
    scores[player] += points
    score_set = set(scores)
    result.append(len(score_set))

for res in result:
    print(res)