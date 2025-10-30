# YOUR CODE HERE
import sys

# Read input from stdin
# Input consists of two space-separated integers A and B on a single line.
line = sys.stdin.readline().split()
A = int(line[0])
B = int(line[1])

# Problem: Find the minimum number of attacks N such that the enemy's stamina becomes 0 or less.
# Initial stamina: A
# Stamina reduced per attack: B
# Stamina after N attacks: A - N * B
# We need to find the smallest integer N such that:
# A - N * B <= 0

# Rearranging the inequality:
# A <= N * B

# Since B is positive (constraint B >= 1), we can divide by B without changing the inequality direction:
# A / B <= N

# We are looking for the smallest integer N that satisfies this condition.
# This is the definition of the ceiling function applied to A / B.
# N = ceil(A / B)

# In integer arithmetic, the ceiling of A / B for positive integers A and B can be calculated as:
# N = (A + B - 1) // B
# where // denotes integer division (floor division).

# Example: A = 7, B = 3
# N = ceil(7 / 3) = ceil(2.33...) = 3
# Formula: (7 + 3 - 1) // 3 = 9 // 3 = 3

# Example: A = 6, B = 3
# N = ceil(6 / 3) = ceil(2) = 2
# Formula: (6 + 3 - 1) // 3 = 8 // 3 = 2

# Calculate the result using the integer division formula.
# Python's integers have arbitrary precision, so they can handle large numbers up to 10^18.
result = (A + B - 1) // B

# Print the result to stdout
print(result)
# YOUR CODE HERE