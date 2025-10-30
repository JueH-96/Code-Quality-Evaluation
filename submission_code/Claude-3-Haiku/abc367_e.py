# Read the input
N, K = map(int, input().split())
X = list(map(int, input().split()))
A = list(map(int, input().split()))

# Perform the operations
for _ in range(K):
    B = [0] * N
    for i in range(N):
        B[i] = A[X[i] - 1]
    A = B

# Print the result
print(" ".join(map(str, A)))