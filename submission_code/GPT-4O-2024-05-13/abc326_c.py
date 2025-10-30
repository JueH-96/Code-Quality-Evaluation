# YOUR CODE HERE
def max_gifts_in_interval(N, M, A):
    A.sort()
    max_gifts = 0
    left = 0
    
    for right in range(N):
        while A[right] - A[left] >= M:
            left += 1
        max_gifts = max(max_gifts, right - left + 1)
    
    return max_gifts

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

print(max_gifts_in_interval(N, M, A))