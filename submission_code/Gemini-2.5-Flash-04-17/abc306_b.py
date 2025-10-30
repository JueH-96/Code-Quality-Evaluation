# YOUR CODE HERE
import sys

# Read the entire line from standard input.
# The line contains 64 space-separated digits (0 or 1).
line = sys.stdin.readline().strip()

# Split the line by spaces into a list of strings.
# Each string in the list corresponds to a digit A_i from A_0 to A_63.
a_str_list = line.split()

# Initialize the variable to store the final sum.
# This variable will accumulate the values of A_i * 2^i.
total = 0

# Iterate through the list of digits. The problem specifies exactly 64 digits.
# The index i in the loop will go from 0 to 63.
# According to the problem statement, the digit at index i (a_str_list[i]) is A_i,
# and it is multiplied by 2^i.
for i in range(64):
    # Convert the string representation of the digit A_i to an integer.
    # Since the constraint is that A_i is 0 or 1, this conversion is straightforward.
    a_i = int(a_str_list[i])

    # If the digit A_i is 1, it contributes 2^i to the total sum.
    # If A_i is 0, it contributes 0 * 2^i = 0, so we don't need to add anything.
    if a_i == 1:
        # Calculate 2 raised to the power of i and add it to the total.
        # Python automatically handles large integers, so calculating 2^i for i up to 63
        # and summing them will not cause overflow issues.
        total += (2 ** i)

# After iterating through all 64 digits and performing the summation,
# the total variable holds the desired result: A_0*2^0 + A_1*2^1 + ... + A_63*2^63.
# Print the final calculated sum to standard output as an integer.
print(total)