# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# dp[j] = maximum points by defeating exactly j monsters
INF = float('inf')
dp = [-INF] * (N + 1)
dp[0] = 0

for i in range(N):
    # Process in reverse order to avoid overwriting values we need
    for j in range(min(i + 1, N), -1, -1):
        if j > 0 and dp[j - 1] != -INF:
            points = A[i]
            if j % 2 == 0:  # j-th defeat (even-numbered)
                points += A[i]
            dp[j] = max(dp[j], dp[j - 1] + points)

print(max(dp))