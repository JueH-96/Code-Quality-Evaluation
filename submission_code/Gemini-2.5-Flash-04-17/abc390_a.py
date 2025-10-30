# YOUR CODE HERE
import sys

# Read the input line from standard input, split by spaces, and convert to integers
line = sys.stdin.readline().split()
a = [int(x) for x in line]

# The target sorted list (1, 2, 3, 4, 5)
sorted_a = [1, 2, 3, 4, 5]

# Flag to indicate if a single adjacent swap can sort the list
found_sortable_swap = False

# Iterate through all possible adjacent swap positions
# For a list of length 5, there are 4 adjacent pairs: (index 0, 1), (1, 2), (2, 3), (3, 4).
# This corresponds to swapping elements at index i and i+1, where i ranges from 0 to 3.
for i in range(4):
    # Create a temporary list that is a copy of the original list 'a'.
    # This is important because we need to try different swaps
    # without permanently changing the original list for the next iteration.
    temp_a = list(a)

    # Perform the adjacent swap of elements at index i and i+1 in the temporary list.
    temp_a[i], temp_a[i+1] = temp_a[i+1], temp_a[i]

    # Check if the list after performing this specific adjacent swap is equal to the sorted list.
    if temp_a == sorted_a:
        # If it is, we have found an adjacent swap that sorts the list.
        # This means the condition is met (it *can* be sorted by exactly one operation).
        found_sortable_swap = True
        # We only need to find one such swap, so we can stop checking further swaps.
        break

# Based on whether we found an adjacent swap that sorts the list, print the result.
if found_sortable_swap:
    print("Yes")
else:
    print("No")