# YOUR CODE HERE
import sys
from collections import defaultdict

def max_gifts(N, M, A):
    A.sort()
    max_count = 0
    left = 0
    
    for right in range(N):
        while A[right] - A[left] >= M:
            left += 1
        max_count = max(max_count, right - left + 1)
    
    return max_count

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

print(max_gifts(N, M, A))