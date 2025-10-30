# YOUR CODE HERE
def find_operations(N, H):
    A = [0] * (N + 1)
    result = [0] * N
    operations = 0
    
    for i in range(N):
        while A[i] <= 0:
            operations += 1
            A[0] += 1
            for j in range(1, N + 1):
                if A[j - 1] > A[j] and A[j - 1] > H[j - 1]:
                    A[j - 1] -= 1
                    A[j] += 1
        result[i] = operations
    
    return result

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
H = list(map(int, data[1:]))

result = find_operations(N, H)
print(" ".join(map(str, result)))