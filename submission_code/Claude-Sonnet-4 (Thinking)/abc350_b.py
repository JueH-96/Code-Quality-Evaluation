# Read input
N, Q = map(int, input().split())
treatments = list(map(int, input().split()))

# Initialize: all holes have teeth
teeth = set(range(1, N + 1))

# Process treatments
for hole in treatments:
    if hole in teeth:
        teeth.remove(hole)
    else:
        teeth.add(hole)

# Count teeth
print(len(teeth))