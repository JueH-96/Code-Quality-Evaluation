# YOUR CODE HERE
N, Q = map(int, input().split())
T = list(map(int, input().split()))

teeth = set(range(1, N + 1))

for t in T:
    if t in teeth:
        teeth.remove(t)
    else:
        teeth.add(t)

print(len(teeth))