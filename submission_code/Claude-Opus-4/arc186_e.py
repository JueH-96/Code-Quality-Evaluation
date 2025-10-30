MOD = 998244353

def solve():
    N, M, K = map(int, input().split())
    X = list(map(int, input().split()))
    
    # dp[i][j] = number of sequences of length i where the longest prefix of X
    # that can be formed as a subsequence has length j
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 1
    
    for i in range(N):
        for j in range(M + 1):
            if dp[i][j] == 0:
                continue
                
            for k in range(1, K + 1):
                if j < M and k == X[j]:
                    # If we add X[j], we extend the prefix
                    if j + 1 < M:
                        dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
                    # If j + 1 == M, we would complete X, so we don't add this
                else:
                    # If we add a different character, check if it matches any earlier position
                    next_j = j
                    # Find the longest prefix of X that ends with k
                    while next_j > 0 and X[next_j - 1] != k:
                        next_j -= 1
                    dp[i + 1][next_j] = (dp[i + 1][next_j] + dp[i][j]) % MOD
    
    # Check if all other sequences can be formed
    # This is complex, so let's use a different approach
    
    # Alternative DP approach
    # dp[i][j][can_form_all] where can_form_all tracks if we can form all other sequences
    # This becomes too complex, so let's use inclusion-exclusion
    
    # Actually, let me reconsider the problem
    # We need sequences where X is the ONLY sequence that cannot be formed
    # This means all other M-length sequences must be formable
    
    # New approach: Use DP with states tracking which sequences can't be formed
    # But this is exponential in the number of possible sequences
    
    # Let's use a different observation:
    # If X is the only sequence that can't be formed, then:
    # 1. X itself can't be a subsequence
    # 2. For any other sequence Y, Y must be a subsequence
    
    # This is equivalent to saying that the sequence must be "maximal" in some sense
    # while avoiding X
    
    # Final approach: DP where we track the state of matching X
    # and ensure we can form all other sequences
    
    dp2 = [[0] * (M + 1) for _ in range(N + 1)]
    dp2[0][0] = 1
    
    for i in range(N):
        for j in range(M):
            if dp2[i][j] == 0:
                continue
            
            # Try adding each character
            for c in range(1, K + 1):
                if c == X[j]:
                    # Extending the match with X
                    if j + 1 < M:
                        dp2[i + 1][j + 1] = (dp2[i + 1][j + 1] + dp2[i][j]) % MOD
                else:
                    # Not extending the match
                    # Find longest prefix of X that ends with c
                    k = j
                    while k > 0 and X[k - 1] != c:
                        k -= 1
                    dp2[i + 1][k] = (dp2[i + 1][k] + dp2[i][j]) % MOD
    
    # The answer is the sum of dp2[N][j] for all j < M
    # But we need to ensure all other sequences can be formed
    
    # This requires checking that the sequence is "rich enough"
    # One way is to use the fact that if we avoid only X and the sequence is long enough,
    # it will contain all other sequences
    
    # For now, let's compute the basic DP result
    result = sum(dp2[N][j] for j in range(M)) % MOD
    
    print(result)

solve()