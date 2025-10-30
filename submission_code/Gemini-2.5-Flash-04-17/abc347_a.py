# YOUR CODE HERE
import sys

# Read N and K from the first line
# N: number of elements in A
# K: the divisor
line1 = sys.stdin.readline().split()
n = int(line1[0])
k = int(line1[1])

# Read the sequence A from the second line
# A: a sequence of N integers, sorted in ascending order
line2 = sys.stdin.readline().split()
a = list(map(int, line2))

# Create a list to store the quotients of elements in A that are multiples of K
result_quotients = []

# Iterate through each element in the sequence A
for x in a:
    # Check if the current element x is a multiple of K
    if x % k == 0:
        # If it is a multiple, calculate the integer quotient x / K
        quotient = x // k
        # Append the quotient to the result list
        result_quotients.append(quotient)

# The original sequence A is sorted in ascending order.
# Since we only extract elements that are multiples of K and divide by a positive K,
# the resulting quotients in result_quotients will also be in ascending order.

# Print the collected quotients
# The problem requires printing them separated by spaces.
# We convert the list of integers to a list of strings using map(str, ...)
# Then join the string elements with a space in between using ' '.join(...)
print(' '.join(map(str, result_quotients)))