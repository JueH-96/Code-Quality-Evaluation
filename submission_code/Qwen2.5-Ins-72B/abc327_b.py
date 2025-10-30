# YOUR CODE HERE
import math

B = int(input())

# Function to check if there exists an integer A such that A^A = B
def find_A(B):
    # The maximum possible value for A is the integer part of the logarithm base 2 of B
    max_A = int(math.log(B, 2)) + 1
    for A in range(1, max_A + 1):
        if A ** A == B:
            return A
    return -1

print(find_A(B))