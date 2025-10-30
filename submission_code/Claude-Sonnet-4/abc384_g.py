# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
K = int(input())

# Precompute the 2D prefix sums
# dp[x][y] = sum of |A_i - B_j| for i from 1 to x and j from 1 to y
dp = [[0] * (N + 1) for _ in range(N + 1)]

for x in range(1, N + 1):
    for y in range(1, N + 1):
        # dp[x][y] = dp[x-1][y] + dp[x][y-1] - dp[x-1][y-1] + sum of |A[x-1] - B[j]| for j from 0 to y-1
        dp[x][y] = dp[x-1][y] + dp[x][y-1] - dp[x-1][y-1]
        for j in range(y):
            dp[x][y] += abs(A[x-1] - B[j])

for _ in range(K):
    x, y = map(int, input().split())
    print(dp[x][y])