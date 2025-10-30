# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))

max_score = max(P)
print(max_score - P[0] + 1 if max_score > P[0] else 0)