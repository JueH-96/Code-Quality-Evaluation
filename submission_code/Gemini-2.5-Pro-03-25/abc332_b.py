# YOUR CODE HERE
import sys

# Read input from stdin
# K: number of operations
# G: capacity of the glass
# M: capacity of the mug
k, g, m = map(int, sys.stdin.readline().split())

# Initialize current water levels in the glass and mug
glass_water = 0
mug_water = 0

# Simulate K operations
for _ in range(k):
    # Check conditions in the specified order as per the problem statement

    # 1. When the glass is filled with water, discard all water from the glass.
    if glass_water == g:
        glass_water = 0

    # 2. Otherwise, if the mug is empty, fill the mug with water.
    elif mug_water == 0:
        mug_water = m

    # 3. Otherwise (glass not full, mug not empty), transfer water from the mug to the glass
    #    until the mug is empty or the glass is filled with water.
    else:
        # Calculate the available space in the glass
        space_in_glass = g - glass_water

        # Determine the amount to transfer: it's the minimum of
        # the water currently in the mug and the available space in the glass.
        transfer_amount = min(mug_water, space_in_glass)

        # Perform the transfer
        glass_water += transfer_amount
        mug_water -= transfer_amount

# Print the final amounts of water in the glass and the mug, separated by a space, to stdout
print(f"{glass_water} {mug_water}")