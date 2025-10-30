MOD = 998244353

def main():
    import sys
    N, *rest = map(int, sys.stdin.read().split())
    A = rest[:N]
    
    product = 1
    for x in A:
        product = product * x % MOD
    
    dp = [0] * 2048
    dp[1] = 1  # initial mask is 1 (sum 0)
    
    for a in A:
        next_dp = [0] * 2048
        for mask in range(2048):
            cnt = dp[mask]
            if cnt == 0:
                continue
            
            # Handle a > 10 terms
            if a > 10:
                count_a_gt = a - 10
            else:
                count_a_gt = 0
            contribution = cnt * count_a_gt
            next_dp[mask] = (next_dp[mask] + contribution) % MOD
            
            # Handle a from 1 to min(a, 10)
            max_a_val = min(a, 10)
            for a_val in range(1, max_a_val + 1):
                shifted = (mask << a_val) & 0x7ff  # 0x7ff is 2047 (11 bits)
                new_mask = mask | shifted
                if (new_mask & (1 << 10)) != 0:
                    continue  # skip if 10 is formed
                next_dp[new_mask] = (next_dp[new_mask] + cnt) % MOD
        
        dp = next_dp
    
    bad = sum(dp) % MOD
    numerator = (product - bad) % MOD
    denominator = product
    ans = numerator * pow(denominator, MOD - 2, MOD) % MOD
    print(ans)
    
if __name__ == "__main__":
    main()