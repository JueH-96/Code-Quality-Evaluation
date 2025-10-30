def lcs_length(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[n][m]

def solve():
    MOD = 998244353
    
    # Read input
    N, M = map(int, input().split())
    S = input().strip()
    
    # For each possible string of length M, we need to count how many have LCS length k
    # We'll use dynamic programming with states
    dp = [[0] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 1
    
    # For each position in the string we're building
    for pos in range(M):
        # For each LCS length achieved so far
        for lcs in range(min(pos + 1, N) + 1):
            if dp[pos][lcs] == 0:
                continue
            
            # Try each possible character
            for c in range(26):
                char = chr(ord('a') + c)
                # Calculate new LCS length after adding this character
                new_lcs = lcs
                # Check if this character can extend LCS
                for i in range(lcs, N):
                    if S[i] == char:
                        new_lcs = lcs + 1
                        break
                
                dp[pos + 1][new_lcs] = (dp[pos + 1][new_lcs] + dp[pos][lcs]) % MOD
    
    # Print answers
    result = []
    for k in range(N + 1):
        result.append(str(dp[M][k]))
    
    print(" ".join(result))

solve()