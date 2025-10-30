# YOUR CODE HERE
N, Q = map(int, input().split())
treatments = list(map(int, input().split()))

# Initially all holes 1 to N have teeth
teeth = set(range(1, N + 1))

# Process each treatment
for hole in treatments:
    if hole in teeth:
        # Remove tooth if present
        teeth.remove(hole)
    else:
        # Add tooth if not present
        teeth.add(hole)

# Output the final count
print(len(teeth))