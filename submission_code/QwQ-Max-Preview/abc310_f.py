MOD = 998244353

def main():
    import sys
    from collections import defaultdict

    n, *rest = map(int, sys.stdin.read().split())
    A = rest[:n]

    total = 1
    for a in A:
        total = (total * a) % MOD

    dp = defaultdict(int)
    dp[0] = 1  # mask 0, has_10=False

    for a in A:
        count_v_less_10 = min(a, 9)
        count_v_greater_10 = max(0, a - 10)
        next_dp = defaultdict(int)

        # Handle v > 10
        if count_v_greater_10 > 0:
            for mask in dp:
                next_dp[mask] = (next_dp[mask] + dp[mask] * count_v_greater_10) % MOD

        # Handle v < 10
        for v in range(1, count_v_less_10 + 1):
            for mask in dp:
                # Check if 10 - v is present in mask
                if (mask & (1 << (10 - v - 1))) != 0:
                    continue  # leads to sum 10, invalid
                # Compute new_mask
                new_mask = mask
                # Add v as a subset sum
                if v <= 9:
                    new_mask |= 1 << (v - 1)
                    # Add s + v for each s in mask where s <=9
                    for s in range(1, 10):
                        if mask & (1 << (s - 1)):
                            sum_sv = s + v
                            if sum_sv <= 9:
                                new_mask |= 1 << (sum_sv - 1)
                next_dp[new_mask] = (next_dp[new_mask] + dp[mask]) % MOD

        dp = next_dp

    invalid = sum(dp.values()) % MOD
    numerator = (total - invalid) % MOD
    denominator_inv = pow(total, MOD - 2, MOD)
    answer = (numerator * denominator_inv) % MOD
    print(answer)

if __name__ == "__main__":
    main()