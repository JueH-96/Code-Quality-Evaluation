# Read input from stdin
A, B = map(int, input().split())

# Calculate the sum of A and B
sum_ab = A + B

# Print any number between 0 and 9 that is not equal to sum_ab
# Using (sum_ab + 1) % 10 ensures a different number in the range
print((sum_ab + 1) % 10)