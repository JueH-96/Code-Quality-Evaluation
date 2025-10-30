# YOUR CODE HERE
N, Q = map(int, input().split())
treatments = list(map(int, input().split()))

# Initially all holes 1 to N have teeth
teeth = set(range(1, N + 1))

# Perform each treatment
for hole in treatments:
    if hole in teeth:
        teeth.remove(hole)  # Remove tooth if present
    else:
        teeth.add(hole)     # Grow tooth if absent

# Print the final number of teeth
print(len(teeth))