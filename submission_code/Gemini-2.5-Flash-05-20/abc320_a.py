# Read the two integers A and B from standard input
# The input will be on a single line, separated by a space
A, B = map(int, input().split())

# Calculate A^B + B^A
# Python's `**` operator performs exponentiation
# Python integers handle arbitrary precision, so large results are fine
result = A**B + B**A

# Print the calculated result
print(result)