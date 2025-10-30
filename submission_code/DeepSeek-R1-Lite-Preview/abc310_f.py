MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Compute x: product of A_i modulo MOD
    x = 1
    for a in A:
        x = (x * a) % MOD
    
    # Initialize DP
    dp = [0] * 11
    dp[0] = 1  # Empty subset
    
    for a in A:
        # To prevent overwriting, we need to make a copy
        new_dp = dp.copy()
        for s in range(10, -1, -1):
            if s >= 1:
                # Include the die, sum over k from 1 to min(a, s)
                add = 0
                for k in range(1, min(a, s) + 1):
                    add = (add + dp[s - k]) % MOD
                new_dp[s] = (new_dp[s] + add) % MOD
            # Exclude the die: new_dp[s] already has dp[s]
        dp = new_dp
    
    y = dp[10]
    
    # Compute z = y * x^{-1} mod MOD
    # Since MOD is prime, x^{-1} = x^{MOD-2} mod MOD
    inv_x = pow(x, MOD - 2, MOD)
    z = (y * inv_x) % MOD
    print(z)

if __name__ == '__main__':
    main()