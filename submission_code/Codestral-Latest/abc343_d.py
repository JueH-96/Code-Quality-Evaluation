import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])

scores = [0] * N
events = []

for i in range(T):
    A = int(data[2 * i + 2]) - 1
    B = int(data[2 * i + 3])
    events.append((A, B))

score_count = defaultdict(int)
unique_scores = 0

for A, B in events:
    if scores[A] == 0:
        unique_scores += 1
    elif score_count[scores[A]] == 1:
        unique_scores -= 1
    score_count[scores[A]] -= 1

    scores[A] += B

    if score_count[scores[A]] == 0:
        unique_scores += 1
    elif score_count[scores[A]] == 1:
        unique_scores += 1
    score_count[scores[A]] += 1

    print(unique_scores)