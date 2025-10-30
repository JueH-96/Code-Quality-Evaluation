def mod_inverse(a, m):
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a, m)
    return (x % m + m) % m

def solve(N, A):
    MOD = 998244353
    total = 1
    for a in A:
        total *= a
    
    # dp[i][s] = number of ways to get sum s using first i dice
    dp = {}
    
    # Initialize with first die
    for v in range(1, A[0]+1):
        dp[(0, v)] = 1
        
    # Process each remaining die
    for i in range(1, N):
        new_dp = {}
        # For each previous sum
        for (prev_i, s), count in dp.items():
            # For each value of current die
            for v in range(1, A[i]+1):
                new_s = s + v
                if new_s <= 10:
                    new_dp[(i, new_s)] = new_dp.get((i, new_s), 0) + count
                
        # Add single die possibilities
        for v in range(1, A[i]+1):
            if v <= 10:
                new_dp[(i, v)] = new_dp.get((i, v), 0) + 1
                
        dp = new_dp
    
    # Count favorable outcomes
    favorable = 0
    for (i, s), count in dp.items():
        if s == 10:
            favorable += count
            
    # Calculate probability
    numerator = favorable
    denominator = total
    
    # Return (numerator * mod_inverse(denominator, MOD)) % MOD
    return (numerator * mod_inverse(denominator, MOD)) % MOD

N = int(input())
A = list(map(int, input().split()))
print(solve(N, A))