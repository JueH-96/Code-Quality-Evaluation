MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    # Initialize DP table: dp[pos][lcs] = count
    dp = [[0]*(N+1) for _ in range(N+1)]
    dp[0][0] = 1
    
    for _ in range(M):
        next_dp = [[0]*(N+1) for _ in range(N+1)]
        for pos in range(N+1):
            for lcs in range(N+1):
                if dp[pos][lcs] == 0:
                    continue
                cnt = dp[pos][lcs]
                if pos < N:
                    s_char = S[pos]
                    # Characters that match S[pos] contribute to new state
                    new_pos1 = pos + 1
                    new_lcs1 = lcs + 1
                    if new_lcs1 <= N:
                        next_dp[new_pos1][new_lcs1] = (next_dp[new_pos1][new_lcs1] + cnt) % MOD
                    # The other 25 characters contribute to staying here
                    next_dp[pos][lcs] = (next_dp[pos][lcs] + 25 * cnt) % MOD
                else:
                    # All characters contribute to staying here
                    next_dp[pos][lcs] = (next_dp[pos][lcs] + 26 * cnt) % MOD
        dp = next_dp
    
    ans = [0]*(N+1)
    for pos in range(N+1):
        for lcs in range(N+1):
            ans[lcs] = (ans[lcs] + dp[pos][lcs]) % MOD
    
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()