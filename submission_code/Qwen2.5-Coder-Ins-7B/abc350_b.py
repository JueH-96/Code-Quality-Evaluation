# YOUR CODE HERE
N, Q = map(int, input().split())
treatment_holes = list(map(int, input().split()))

teeth = set(range(1, N + 1))

for hole in treatment_holes:
    if hole in teeth:
        teeth.remove(hole)
    else:
        teeth.add(hole)

print(len(teeth))