# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# dp[i][j] = number of ways to match first j elements of B using first i elements of A
dp = [[0] * (M + 1) for _ in range(N + 1)]

# Base case: empty subsequence can be formed in 1 way
for i in range(N + 1):
    dp[i][0] = 1

# Fill the dp table
for i in range(1, N + 1):
    for j in range(1, M + 1):
        # We can always skip A[i-1]
        dp[i][j] = dp[i-1][j]
        
        # If A[i-1] matches B[j-1], we can also use it
        if A[i-1] == B[j-1]:
            dp[i][j] += dp[i-1][j-1]

# Check if there are at least 2 ways to form B
if dp[N][M] >= 2:
    print("Yes")
else:
    print("No")