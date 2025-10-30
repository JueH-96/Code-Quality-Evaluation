# Read input values
N, Q = map(int, input().split())
T = list(map(int, input().split()))

# Initialize the set of teeth
teeth = set(range(1, N+1))

# Perform the treatments
for t in T:
    if t in teeth:
        teeth.remove(t)
    else:
        teeth.add(t)

# Output the number of teeth
print(len(teeth))