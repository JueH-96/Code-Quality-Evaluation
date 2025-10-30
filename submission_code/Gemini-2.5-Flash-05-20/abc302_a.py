# YOUR CODE HERE
# Read the two integers A and B from a single line of input.
# input().split() reads a line and splits it into a list of strings.
# map(int, ...) converts each string in the list to an integer.
A, B = map(int, input().split())

# Calculate the minimum number of attacks using the ceiling division formula.
# For positive integers A and B, ceil(A / B) can be computed as (A + B - 1) // B.
# This formula correctly handles both cases where A is a multiple of B and when it's not.
# For example:
# If A = 7, B = 3: (7 + 3 - 1) // 3 = 9 // 3 = 3
# If A = 6, B = 3: (6 + 3 - 1) // 3 = 8 // 3 = 2
result = (A + B - 1) // B

# Print the calculated result.
print(result)