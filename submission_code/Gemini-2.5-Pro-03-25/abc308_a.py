import sys

# Read the 8 integers from standard input
# Input is guaranteed to be 8 integers on a single line separated by spaces.
s = list(map(int, sys.stdin.readline().split()))

# Flag to track if all conditions are satisfied. Initialize to True.
all_conditions_met = True

# Iterate through each element S_i using its index i from 0 to 7.
for i in range(8):
    # Check Condition 2: Each S_i must be between 100 and 675, inclusive.
    if not (100 <= s[i] <= 675):
        all_conditions_met = False
        # If condition fails, no need to check further. Exit the loop.
        break

    # Check Condition 3: Each S_i must be a multiple of 25.
    # This is equivalent to checking if S_i modulo 25 is 0.
    if s[i] % 25 != 0:
        all_conditions_met = False
        # If condition fails, no need to check further. Exit the loop.
        break

    # Check Condition 1: The sequence must be monotonically non-decreasing.
    # This means S_i must be greater than or equal to S_{i-1}.
    # This check is only needed for elements starting from the second one (i > 0).
    if i > 0:
        # Compare the current element s[i] with the previous element s[i-1].
        if s[i-1] > s[i]:
            all_conditions_met = False
            # If condition fails, no need to check further. Exit the loop.
            break

# After iterating through all elements (or breaking early if a condition failed),
# check the final status of the flag.
if all_conditions_met:
    # If the flag is still True, all conditions were met for all elements.
    print("Yes")
else:
    # If the flag is False, at least one condition failed.
    print("No")