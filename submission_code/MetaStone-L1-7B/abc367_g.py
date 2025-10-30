MOD = 998244353

def main():
    import sys
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    dp = [{} for _ in range(M)]
    dp[0][0] = 1  # Initially, one subsequence (empty) with XOR 0 and count 1
    
    for a in A:
        a_val = a
        new_dp = [{} for _ in range(M)]
        for r in range(M):
            current = dp[r]
            if not current:
                continue
            # The number of existing subsequences is the sum of counts
            for x in current:
                cnt = current[x]
                # When not including a, the state remains
                if x in new_dp[r]:
                    new_dp[r][x] = (new_dp[r][x] + cnt) % MOD
                else:
                    new_dp[r][x] = cnt % MOD
                # When including a, the new XOR is x ^ a_val, and the new length is (r+1) % M
                r_next = (r + 1) % M
                if (x ^ a_val) in new_dp[r_next]:
                    new_dp[r_next][x ^ a_val] = (new_dp[r_next][x ^ a_val] + cnt) % MOD
                else:
                    new_dp[r_next][x ^ a_val] = cnt % MOD
        dp = new_dp
    
    result = 0
    for r in range(M):
        if M == 0:
            continue
        if r == 0:
            for x in dp[r]:
                result = (result + pow(x, K, MOD)) % MOD
    print(result % MOD)

if __name__ == '__main__':
    main()