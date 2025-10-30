# YOUR CODE HERE
import math
from collections import defaultdict

def is_square(n):
    if n < 0:
        return False
    root = int(math.isqrt(n))
    return root * root == n

def count_square_pairs(N, A):
    count = 0
    freq = defaultdict(int)
    
    for num in A:
        freq[num] += 1
    
    for i in range(N):
        for j in range(i + 1, N):
            if is_square(A[i] * A[j]):
                count += 1
    
    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(count_square_pairs(N, A))