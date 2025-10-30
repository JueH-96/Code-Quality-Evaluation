import sys

# Read the input line
line = sys.stdin.readline().split()

# Convert the input strings to integers
A = int(line[0])
B = int(line[1])

# The problem is to find the minimum integer k such that A - k * B <= 0.
# This inequality can be rewritten as k * B >= A.
# Since B is positive, we can divide by B: k >= A / B.
# We are looking for the smallest integer k that is greater than or equal to A / B.
# This is the ceiling of A / B.
# For positive integers A and B, the ceiling of A / B can be calculated using integer division as (A + B - 1) // B.

attacks = (A + B - 1) // B

# Print the result
print(attacks)