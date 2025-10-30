# YOUR CODE HERE
import sys

# Read input values for K, G, and M
# K: number of operations
# G: capacity of the glass
# M: capacity of the mug
K, G, M = map(int, sys.stdin.readline().split())

# Initialize the current amounts of water in the glass and the mug
glass = 0
mug = 0

# Simulate the operations K times
for _ in range(K):
    # Check the rules in the specified order

    # Rule 1: When the glass is filled with water, discard all the water from the glass.
    if glass == G:
        glass = 0
    # Rule 2: Otherwise, if the mug is empty, fill the mug with water.
    elif mug == 0:
        mug = M
    # Rule 3: Otherwise, transfer water from the mug to the glass
    # until the mug is empty or the glass is filled with water.
    else:
        # The amount of water to transfer is limited by two factors:
        # 1. The remaining capacity in the glass (G - glass)
        # 2. The amount of water currently in the mug (mug)
        transfer_amount = min(G - glass, mug)

        # Perform the transfer
        glass += transfer_amount
        mug -= transfer_amount

# After K operations, print the final amounts of water in the glass and the mug
print(glass, mug)