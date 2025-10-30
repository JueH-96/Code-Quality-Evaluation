N, M = map(int, input().split())
S = input().strip()

MOD = 998244353

# dp[i][j][k] = number of strings T[0:j] such that LCS(S[0:i], T[0:j]) = k
dp = [[[0] * (N + 1) for _ in range(M + 1)] for _ in range(N + 1)]

# Base cases
for j in range(M + 1):
    dp[0][j][0] = pow(26, j, MOD)

for i in range(N + 1):
    dp[i][0][0] = 1

for i in range(1, N + 1):
    for j in range(1, M + 1):
        for k in range(N + 1):
            # Case 1: T[j-1] = S[i-1]
            if k > 0:
                dp[i][j][k] = (dp[i][j][k] + dp[i-1][j-1][k-1]) % MOD
            
            # Case 2: T[j-1] != S[i-1] (25 choices)
            for k2 in range(k + 1):  # LCS(S[0:i], T[0:j-1])
                if dp[i][j-1][k2] == 0:
                    continue
                    
                # When we add a non-matching character, 
                # LCS(S[0:i], T[0:j]) = max(LCS(S[0:i-1], T[0:j-1]), k2)
                # For this to equal k, we need max(something, k2) = k
                
                if k2 == k:
                    # If LCS(S[0:i], T[0:j-1]) = k, then adding any non-matching char keeps it k
                    dp[i][j][k] = (dp[i][j][k] + 25 * dp[i][j-1][k2]) % MOD
                elif k2 < k:
                    # Need LCS(S[0:i-1], T[0:j-1]) = k for the max to be k
                    # Add contribution from strings where this condition holds
                    dp[i][j][k] = (dp[i][j][k] + 25 * dp[i-1][j-1][k]) % MOD

for k in range(N + 1):
    print(dp[N][M][k], end=' ' if k < N else '
')