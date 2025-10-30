# YOUR CODE HERE

import sys

def min_score(N, X, scores):
    scores.sort()
    min_score = scores[1]
    total = sum(scores[1:-1])
    while total < X:
        min_score += 1
        total += 1
    return min_score if total <= X else -1

N, X = map(int, sys.stdin.readline().split())
scores = list(map(int, sys.stdin.readline().split()))
print(min_score(N, X, scores))