# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
scores = list(map(int, sys.stdin.readline().strip().split()))

max_score = max(scores)
scores.remove(max_score)

for i in range(1, max_score):
    if all(score + i > max_score for score in scores):
        print(i)
        break