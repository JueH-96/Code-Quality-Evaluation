# YOUR CODE HERE
import math

def find_A(B):
    # Check for small values of A first
    for A in range(1, int(math.isqrt(B)) + 1):
        if A ** A == B:
            return A
    return -1

# Read input
B = int(input().strip())

# Find and print the result
result = find_A(B)
print(result)