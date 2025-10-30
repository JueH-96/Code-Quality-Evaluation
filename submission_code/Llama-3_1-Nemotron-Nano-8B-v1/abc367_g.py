MOD = 998244353

def main():
    import sys
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Precompute x^k mod MOD for all possible x (up to 2^20)
    max_xor = 1 << 20
    pow_x = [pow(x, K, MOD) for x in range(max_xor)]
    
    # DP table: dp[r][x] = number of subsets with size ≡ r mod M and xor x
    dp = [dict() for _ in range(M)]
    dp[0][0] = 1  # empty subset
    
    for a in A:
        new_dp = [dict(dpr) for drp in dp]  # copy for not taking a
        for r in range(M):
            for x in list(dp[r].keys()):
                cnt = dp[r][x]
                new_r = (r + 1) % M
                new_x = x ^ a
                if new_x in new_dp[new_r]:
                    new_dp[new_r][new_x] = (new_dp[new_r][new_x] + cnt) % MOD
                else:
                    new_dp[new_r][new_x] = cnt % MOD
        dp = new_dp
    
    # Compute the answer: sum over x of dp[0][x] * x^k
    ans = 0
    for x, cnt in dp[0].items():
        if x == 0 and cnt == 1:
            continue  # exclude empty subset
        ans = (ans + pow_x[x] * cnt) % MOD
    print(ans % MOD)

if __name__ == "__main__":
    main()