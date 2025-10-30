# YOUR CODE HERE
N = int(input().strip())
A = list(map(int, input().strip().split()))
B = list(map(int, input().strip().split()))
K = int(input().strip())

# Pre-compute prefix sums for each row
prefix_sum = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, N + 1):
        prefix_sum[i][j] = prefix_sum[i][j-1] + abs(A[i-1] - B[j-1])

# Pre-compute the result for all possible values of (X_k, Y_k)
result = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, N + 1):
        result[i][j] = result[i-1][j] + prefix_sum[i][j]

for _ in range(K):
    X, Y = map(int, input().strip().split())
    print(result[X][Y])