# YOUR CODE HERE
import sys

# Read the input line from standard input. 
# The line contains 64 space-separated integers (0 or 1).
line = sys.stdin.readline()

# Split the line into a list of strings based on spaces.
# Each string in the list 'parts' represents one of A_0, A_1, ..., A_63.
parts = line.split()

# Convert the list of strings into a list of integers.
# The list 'a' will store the sequence A = (A_0, A_1, ..., A_63).
# Specifically, a[i] will hold the value of A_i.
a = [int(x) for x in parts]

# Initialize the sum to 0. This variable will accumulate the final result.
total_sum = 0

# The problem asks for the sum A_0 * 2^0 + A_1 * 2^1 + ... + A_63 * 2^{63}.
# We can compute this sum by iterating through the indices i from 0 to 63.
for i in range(64):
    # For each index i, check if the coefficient A_i (stored in a[i]) is 1.
    # If A_i is 0, the term A_i * 2^i is 0, so it doesn't contribute to the sum.
    if a[i] == 1:
        # If A_i is 1, the term is 1 * 2^i = 2^i.
        # We add this value to our running total sum.
        # The expression (1 << i) is a bitwise left shift operation, which is an
        # efficient way to calculate 2 to the power of i (2**i).
        # Python's integers handle arbitrary size, so we don't need to worry about overflow
        # for values up to 2^63.
        total_sum += (1 << i)

# After iterating through all the coefficients from A_0 to A_63,
# 'total_sum' holds the final calculated value.
# Print the final computed sum to standard output.
print(total_sum)