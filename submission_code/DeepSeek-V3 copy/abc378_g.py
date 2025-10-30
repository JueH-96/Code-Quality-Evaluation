# YOUR CODE HERE
import sys
import math

def count_permutations(A, B, M):
    N = A * B - 1
    # The number of such permutations is the number of standard Young tableaux of shape (A, A, ..., A) (B times)
    # which is given by the hook-length formula
    # The shape is a rectangle with A rows and B columns
    # The hook lengths are:
    # For each cell (i,j), the hook length is (A - i) + (B - j) + 1
    # The total number of tableaux is N! / (product of hook lengths)
    
    # Compute N!
    factorial = 1
    for i in range(1, N+1):
        factorial = (factorial * i) % M
    
    # Compute the product of hook lengths
    hook_product = 1
    for i in range(A):
        for j in range(B):
            hook_length = (A - i) + (B - j) - 1
            hook_product = (hook_product * hook_length) % M
    
    # Compute the modular inverse of hook_product modulo M
    # Since M is prime, we can use Fermat's little theorem
    inv_hook_product = pow(hook_product, M-2, M)
    
    # The number of tableaux is (N! * inv_hook_product) % M
    result = (factorial * inv_hook_product) % M
    return result

# Read input
A, B, M = map(int, sys.stdin.readline().split())

# Compute the result
result = count_permutations(A, B, M)

# Print the result
print(result)