# Read the input values
K, G, M = map(int, input().split())

# Initialize the glass and mug to 0 milliliters
glass = 0
mug = 0

for _ in range(K):
    if glass == G:
        glass = 0  # Discard all water from the glass
    else:
        if mug == 0:
            mug = M  # Fill the mug if it's empty
        else:
            # Transfer as much as possible from mug to glass
            transfer = min(mug, G - glass)
            glass += transfer
            mug -= transfer

# Output the final amounts in the glass and mug
print(glass, mug)