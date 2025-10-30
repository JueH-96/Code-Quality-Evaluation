# YOUR CODE HERE

N, Q = map(int, input().split())
teeth = [0] * (N + 1)

for _ in range(Q):
    T = list(map(int, input().split()))
    for t in T:
        teeth[t] = 1

print(sum(teeth))