# YOUR CODE HERE
import math

def find_A(B):
    if B == 1:
        return 1
    for A in range(2, int(math.log(B, 2)) + 2):
        if A ** A == B:
            return A
    return -1

# Read input
B = int(input().strip())

# Find and print the result
print(find_A(B))