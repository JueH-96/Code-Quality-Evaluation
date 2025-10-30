# YOUR CODE HERE
N, Q = map(int, input().split())
treatments = list(map(int, input().split()))

# Initially all holes have teeth
teeth = set(range(1, N + 1))

# Process each treatment
for hole in treatments:
    if hole in teeth:
        teeth.remove(hole)  # Remove tooth
    else:
        teeth.add(hole)     # Add tooth

# Print the final count
print(len(teeth))