# YOUR CODE HERE
def can_divide_into_equal_sums(A, B, C):
    # Check if any two numbers sum up to the third one
    if A + B == C or A + C == B or B + C == A:
        return "Yes"
    return "No"

# Read input
import sys
input = sys.stdin.read().strip()
A, B, C = map(int, input.split())

# Determine if it's possible to divide into equal sums
result = can_divide_into_equal_sums(A, B, C)

# Output the result
print(result)