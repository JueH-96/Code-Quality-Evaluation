# Read the inputs from stdin
K, G, M = map(int, input().split())

# Initialize the glass and mug capacities
glass = 0
mug = 0

# Perform the operation K times
for _ in range(K):
    if glass == G:  # If the glass is full, empty it
        glass = 0
    elif mug == 0:  # If the mug is empty, fill it
        mug = M
    else:  # Transfer water from the mug to the glass
        transfer = min(G - glass, mug)
        glass += transfer
        mug -= transfer

# Output the result
print(glass, mug)