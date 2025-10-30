# YOUR CODE HERE
def find_max_sum(N, M, D, A, B):
    A.sort()
    B.sort()
    
    max_sum = -1
    j = 0
    
    for i in range(N):
        while j < M and B[j] < A[i] - D:
            j += 1
        if j < M and B[j] <= A[i] + D:
            max_sum = max(max_sum, A[i] + B[j])
    
    return max_sum

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
D = int(data[2])
A = list(map(int, data[3:3+N]))
B = list(map(int, data[3+N:]))

print(find_max_sum(N, M, D, A, B))