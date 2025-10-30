# YOUR CODE HERE
def find_columns(A):
    # We need to find B such that A * B = 400
    # This means B = 400 / A should be an integer
    if 400 % A == 0:
        return 400 // A
    else:
        return -1

# Read input
import sys
input = sys.stdin.read
A = int(input().strip())

# Find and print the result
result = find_columns(A)
print(result)