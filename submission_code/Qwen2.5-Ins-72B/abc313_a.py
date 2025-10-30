# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))

max_score = max(P[1:])
if P[0] >= max_score:
    print(0)
else:
    print(max_score - P[0] + 1)