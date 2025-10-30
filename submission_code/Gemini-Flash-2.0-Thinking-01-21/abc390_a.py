import sys

# Read the five integers from standard input
line = sys.stdin.readline().split()
a = [int(x) for x in line]

# The target sorted sequence
target = [1, 2, 3, 4, 5]

# Generate all sequences that can be sorted into the target sequence
# by performing exactly one adjacent swap.
# These are sequences obtained by performing one adjacent swap on the target sequence.
one_swap_sortable_sequences = []

# Iterate through possible adjacent swap positions (indices 0, 1, 2, 3 for a list of length 5)
# For a list of length N, adjacent swap indices are (0,1), (1,2), ..., (N-2, N-1)
# This corresponds to starting indices i from 0 to N-2.
for i in range(len(target) - 1):
    # Create a temporary copy of the target sequence
    temp_target = list(target)
    # Swap the element at index i with the element at index i+1
    temp_target[i], temp_target[i+1] = temp_target[i+1], temp_target[i]
    # Add the resulting sequence to our list of possible inputs
    one_swap_sortable_sequences.append(temp_target)

# Check if the input sequence 'a' is present in the list of sortable sequences
if a in one_swap_sortable_sequences:
    # If it is, it means 'a' can be sorted by exactly one adjacent swap
    print("Yes")
else:
    # If it's not, it means 'a' cannot be sorted by exactly one adjacent swap
    # (either it's already sorted, or requires more than one swap, etc.)
    print("No")