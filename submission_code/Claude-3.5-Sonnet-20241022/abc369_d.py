N = int(input())
A = list(map(int, input().split()))

# dp[i][j] represents max points when considering first i monsters 
# and having defeated j monsters so far
dp = [[-1]*(N+1) for _ in range(N+1)]
dp[0][0] = 0

for i in range(N):
    for j in range(i+1):
        if dp[i][j] == -1:
            continue
            
        # Don't defeat current monster
        dp[i+1][j] = max(dp[i+1][j], dp[i][j])
        
        # Defeat current monster
        points = A[i]
        if (j+1) % 2 == 0:  # Even numbered defeat
            points *= 2
        dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j] + points)

print(max(dp[N]))