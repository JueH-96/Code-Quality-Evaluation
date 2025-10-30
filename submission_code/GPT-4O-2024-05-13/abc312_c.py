# YOUR CODE HERE
def find_minimum_X(N, M, A, B):
    A.sort()
    B.sort()
    
    left, right = 1, max(max(A), max(B)) + 1
    
    while left < right:
        mid = (left + right) // 2
        
        sellers_count = sum(1 for a in A if a <= mid)
        buyers_count = sum(1 for b in B if b >= mid)
        
        if sellers_count >= buyers_count:
            right = mid
        else:
            left = mid + 1
    
    return left

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:]))

print(find_minimum_X(N, M, A, B))