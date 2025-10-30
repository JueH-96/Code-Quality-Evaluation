# YOUR CODE HERE
import sys

# Read the two integers from standard input.
# The input line contains two space-separated integers A and B.
# map(int, ...) applies the int function to each element of the list returned by split().
# The result of map is unpacked into variables A and B.
A, B = map(int, sys.stdin.readline().split())

# Calculate the sum of A and B.
sum_ab = A + B

# Calculate the square of the sum.
result = sum_ab ** 2

# Print the result to standard output.
print(result)