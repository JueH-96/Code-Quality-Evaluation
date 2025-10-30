# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:]))

def min_operations(A, B, M):
    operations = 0
    for i in range(N):
        if A[i] != B[i]:
            diff = abs(A[i] - B[i])
            if diff == 1 or diff == M - 1:
                operations += 1
            else:
                operations += min(diff, M - diff)
    return operations

result = min_operations(A, B, M)
print(result)