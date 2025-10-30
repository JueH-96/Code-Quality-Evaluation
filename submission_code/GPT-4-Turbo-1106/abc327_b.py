import math

# Read the input from stdin
B = int(input())

# Initialize the result to -1, assuming no such A exists
result = -1

# The maximum possible value for A can be found by taking the B-th root of B.
# Since B is at most 10^18, we can safely start with A = B^(1/B) and decrease A until A^A < B.
# We use logarithms to avoid floating point errors.
max_possible_A = int(B ** (1 / math.log(B, math.e)))

# Iterate from max_possible_A down to 1 to find a possible A
for A in range(max_possible_A, 0, -1):
    if A ** A == B:
        result = A
        break

# Print the result
print(result)