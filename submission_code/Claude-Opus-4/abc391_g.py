# YOUR CODE HERE
MOD = 998244353

def count_lcs_at_least_k(S, M, k):
    N = len(S)
    if k > N or k > M:
        return 0
    
    # dp[i][j][matched] = number of strings of length j where:
    # - we've considered first i characters of S
    # - we've matched exactly 'matched' characters from S
    dp = [[[0 for _ in range(k+1)] for _ in range(M+1)] for _ in range(N+1)]
    
    # Base case: empty string S, empty target string
    dp[0][0][0] = 1
    
    for i in range(N+1):
        for j in range(M+1):
            for matched in range(min(i, j, k) + 1):
                if dp[i][j][matched] == 0:
                    continue
                
                # If we've filled all M positions
                if j == M:
                    continue
                
                # Case 1: Add any character that doesn't match S[i] (if i < N)
                if i < N:
                    # 25 choices (26 - 1 for S[i])
                    dp[i][j+1][matched] = (dp[i][j+1][matched] + dp[i][j][matched] * 25) % MOD
                else:
                    # All 26 choices if we've exhausted S
                    dp[i][j+1][matched] = (dp[i][j+1][matched] + dp[i][j][matched] * 26) % MOD
                
                # Case 2: Add character that matches S[i] and advance in S
                if i < N and matched < k:
                    dp[i+1][j+1][matched+1] = (dp[i+1][j+1][matched+1] + dp[i][j][matched]) % MOD
                
                # Case 3: Skip S[i] without matching
                if i < N:
                    dp[i+1][j][matched] = (dp[i+1][j][matched] + dp[i][j][matched]) % MOD
    
    # Sum all ways to have exactly k matches
    result = 0
    for i in range(N+1):
        result = (result + dp[i][M][k]) % MOD
    
    return result

N, M = map(int, input().split())
S = input().strip()

# For each k, calculate strings with LCS exactly k
results = []

for k in range(N+1):
    # Count strings with LCS >= k
    at_least_k = count_lcs_at_least_k(S, M, k)
    
    # Count strings with LCS >= k+1
    at_least_k_plus_1 = count_lcs_at_least_k(S, M, k+1)
    
    # Strings with LCS exactly k = (LCS >= k) - (LCS >= k+1)
    exactly_k = (at_least_k - at_least_k_plus_1 + MOD) % MOD
    results.append(exactly_k)

print(' '.join(map(str, results)))