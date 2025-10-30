# YOUR CODE HERE

N, Q = map(int, input().split())
T = list(map(int, input().split()))

teeth = set()

for i in range(Q):
    if T[i] in teeth:
        teeth.remove(T[i])
    else:
        teeth.add(T[i])

print(len(teeth))