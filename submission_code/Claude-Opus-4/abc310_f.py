MOD = 998244353

def modinv(a, m=MOD):
    # Extended Euclidean Algorithm for finding modular inverse
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a, m)
    return (x % m + m) % m

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # Total number of outcomes
    total = 1
    for ai in a:
        total = (total * ai) % MOD
    
    # Count outcomes where no subset sums to 10
    # dp[mask][achievable_sums] = number of ways
    # Since we only care about sums up to 10, we can use a bitmask for achievable sums
    
    # dp[i][mask] = number of ways to assign values to first i dice
    # such that the achievable sums are represented by mask
    # (bit j is set if sum j is achievable)
    
    dp = [[0] * (1 << 11) for _ in range(n + 1)]
    dp[0][1] = 1  # Initially, only sum 0 is achievable
    
    for i in range(n):
        for mask in range(1 << 11):
            if dp[i][mask] == 0:
                continue
            
            # Try each possible value for die i
            for val in range(1, a[i] + 1):
                new_mask = mask
                # Update achievable sums
                for s in range(11):
                    if mask & (1 << s):
                        if s + val <= 10:
                            new_mask |= (1 << (s + val))
                
                dp[i + 1][new_mask] = (dp[i + 1][new_mask] + dp[i][mask]) % MOD
    
    # Count outcomes where sum 10 is not achievable
    no_sum_10 = 0
    for mask in range(1 << 11):
        if not (mask & (1 << 10)):  # Sum 10 is not achievable
            no_sum_10 = (no_sum_10 + dp[n][mask]) % MOD
    
    # Favorable outcomes = Total - No sum 10
    favorable = (total - no_sum_10 + MOD) % MOD
    
    # Calculate probability as favorable/total mod MOD
    result = (favorable * modinv(total)) % MOD
    print(result)

solve()