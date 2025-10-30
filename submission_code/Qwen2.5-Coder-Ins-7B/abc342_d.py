# YOUR CODE HERE
import sys
from collections import defaultdict

def is_square(n):
    return int(n**0.5)**2 == n

def count_square_pairs(N, A):
    count = 0
    square_counts = defaultdict(int)
    
    for i in range(N):
        for j in range(i + 1, N):
            product = A[i] * A[j]
            if is_square(product):
                count += 1
    
    return count

# Read input
N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

# Solve the problem
result = count_square_pairs(N, A)

# Print the result
print(result)