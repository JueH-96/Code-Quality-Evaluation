# YOUR CODE HERE
def min_operations_to_transform(N, M, A, B):
    def mod_diff(a, b, M):
        return min((b - a) % M, (a - b) % M)

    total_operations = 0
    for i in range(N):
        if A[i] != B[i]:
            diff = mod_diff(A[i], B[i], M)
            total_operations += diff
            A[i] = B[i]
            if i > 0 and A[i] == A[i-1]:
                return -1
            if i < N-1 and A[i] == A[i+1]:
                return -1

    return total_operations

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:2*N+2]))

result = min_operations_to_transform(N, M, A, B)
print(result)