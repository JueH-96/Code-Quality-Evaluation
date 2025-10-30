MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    total = 1
    for a in A:
        total = (total * a) % MOD
    
    size = 1 << 11  # 2048
    dp = [0] * size
    dp[1] = 1  # Initially, only subset sum 0 is possible
    
    for a in A:
        new_dp = [0] * size
        for mask in range(size):
            if dp[mask] == 0:
                continue
            cnt = dp[mask]
            # Handle values greater than 10
            allowed_gt = max(0, a - 10)
            new_dp[mask] = (new_dp[mask] + cnt * allowed_gt) % MOD
            # Handle values from 1 to min(10, a)
            max_v = min(10, a)
            for v in range(1, max_v + 1):
                if v == 10:
                    continue
                # Check if adding v would create a subset sum of 10
                if (mask & (1 << (10 - v))):
                    continue
                # Compute new_mask
                shifted = mask << v
                shifted_mask = shifted & ((1 << 11) - 1)
                new_mask = mask | shifted_mask
                new_dp[new_mask] = (new_dp[new_mask] + cnt) % MOD
        dp = new_dp
    
    # Calculate invalid outcomes
    invalid = 0
    for mask in range(size):
        if not (mask & (1 << 10)):
            invalid = (invalid + dp[mask]) % MOD
    
    # Calculate valid and the answer
    valid = (total - invalid) % MOD
    ans = valid * pow(total, MOD - 2, MOD) % MOD
    print(ans)

if __name__ == "__main__":
    main()