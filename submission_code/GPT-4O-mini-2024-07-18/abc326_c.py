def max_gifts(N, M, A):
    A.sort()
    max_count = 0
    j = 0
    
    for i in range(N):
        while j < N and A[j] < A[i] + M:
            j += 1
        max_count = max(max_count, j - i)
    
    return max_count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

result = max_gifts(N, M, A)
print(result)