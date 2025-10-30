import sys

N, X = map(int, input().split())
S = list(map(int, input().split()))

total_score = sum(score for score in S if score <= X)
print(total_score)