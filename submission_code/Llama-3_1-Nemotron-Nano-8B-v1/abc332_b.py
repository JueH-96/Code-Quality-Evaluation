# Read input values
K, G, M = map(int, input().split())

# Initialize the glass and mug
glass = 0
mug = 0

# Perform K operations
for _ in range(K):
    if glass == G:
        glass = 0
    else:
        if mug == 0:
            mug = M
        else:
            # Calculate the amount to transfer
            transfer = min(mug, G - glass)
            glass += transfer
            mug -= transfer

# Output the result
print(glass, mug)