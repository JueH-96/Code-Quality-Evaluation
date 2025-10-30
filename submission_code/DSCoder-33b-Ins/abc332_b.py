# YOUR CODE HERE

# Read the inputs
K, G, M = map(int, input().split())

# Initialize the glass and mug with 0 milliliters
glass = 0
mug = M

# Perform the operation K times
for _ in range(K):
    # If the glass is filled with water, discard all the water from the glass
    if glass == G:
        glass = 0
    # If the mug is empty, fill the mug with water
    elif mug == 0:
        mug = M
    # Otherwise, transfer water from the mug to the glass until the mug is empty or the glass is filled
    else:
        # Calculate the amount of water that can be transferred
        transfer = min(G - glass, mug)
        # Transfer the water
        glass += transfer
        mug -= transfer

# Print the amounts of water in the glass and the mug
print(glass, mug)