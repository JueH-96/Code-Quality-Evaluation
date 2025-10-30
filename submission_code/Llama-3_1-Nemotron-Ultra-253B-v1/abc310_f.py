MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    dp = [0] * (1 << 10)
    dp[1 << 0] = 1  # Initial state: sum 0
    
    for a in A:
        new_dp = [0] * (1 << 10)
        for mask in range(1 << 10):
            if dp[mask] == 0:
                continue
            # Compute X_success for this mask
            X_success = set()
            for s in range(10):
                if (mask >> s) & 1:
                    x = 10 - s
                    if 1 <= x <= a:
                        X_success.add(x)
            c = len(X_success)
            # Probability of success
            success_prob = dp[mask] * c % MOD
            # Add to the success probability
            new_dp[0] = (new_dp[0] + success_prob) % MOD  # 0 is a dummy state for success
            
            # Process remaining x's
            if a - c == 0:
                continue
            # Compute new_mask
            new_mask = mask
            for s in range(10):
                if (mask >> s) & 1:
                    max_x = min(a, 9 - s)
                    if max_x < 1:
                        continue
                    # Count x's not in X_success in 1..max_x
                    cnt = max_x - sum(1 for x in X_success if 1 <= x <= max_x)
                    if cnt <= 0:
                        continue
                    # Add all possible s + x to new_mask
                    for x in range(1, max_x + 1):
                        if x in X_success:
                            continue
                        t = s + x
                        if t >= 10:
                            continue
                        new_mask |= 1 << t
            # Update new_dp
            new_dp[new_mask] = (new_dp[new_mask] + dp[mask] * (a - c) % MOD) % MOD
        
        dp = new_dp
    
    # The answer is the success probability divided by the total possible, but the code computes it as new_dp[0]
    # However, the code needs to compute the success probability as new_dp[0] divided by the product of A_i
    # Compute the denominator
    den = 1
    for a in A:
        den = den * a % MOD
    inv_den = pow(den, MOD-2, MOD)
    answer = new_dp[0] * inv_den % MOD
    print(answer)

if __name__ == '__main__':
    main()