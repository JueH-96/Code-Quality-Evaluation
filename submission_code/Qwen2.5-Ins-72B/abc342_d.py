import math
from collections import defaultdict

def is_square(n):
    if n == 0:
        return True
    root = int(math.isqrt(n))
    return root * root == n

def count_square_pairs(N, A):
    count = 0
    freq = defaultdict(int)
    
    for i in range(N):
        for j in range(i + 1, N):
            if is_square(A[i] * A[j]):
                count += 1
        freq[A[i]] += 1
    
    # Count pairs where both elements are zero
    zero_pairs = freq[0] * (freq[0] - 1) // 2
    count += zero_pairs
    
    return count

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve and print the result
print(count_square_pairs(N, A))