MOD = 998244353

def modinv(a, m=MOD):
    return pow(a, m-2, m)

def solve():
    n = int(input())
    A = list(map(int, input().split()))
    
    # Use DP to count outcomes where subset sum 10 is possible
    # dp[i][mask] where mask represents achievable sums (up to 10)
    
    # Since we only care about sums up to 10, we can use bitmask
    dp = [[0] * (1 << 11) for _ in range(n + 1)]
    dp[0][1] = 1  # Initially only sum 0 is achievable (bit 0 set)
    
    for i in range(n):
        for mask in range(1 << 11):
            if dp[i][mask] == 0:
                continue
            
            # For each possible value of die i+1
            for val in range(1, A[i] + 1):
                new_mask = mask
                # Add new achievable sums
                for s in range(11):
                    if (mask & (1 << s)) and s + val <= 10:
                        new_mask |= (1 << (s + val))
                
                dp[i + 1][new_mask] = (dp[i + 1][new_mask] + dp[i][mask]) % MOD
    
    # Count favorable outcomes (where sum 10 is achievable)
    favorable = 0
    for mask in range(1 << 11):
        if mask & (1 << 10):  # Sum 10 is achievable
            favorable = (favorable + dp[n][mask]) % MOD
    
    # Total outcomes
    total = 1
    for a in A:
        total = (total * a) % MOD
    
    # Result
    result = (favorable * modinv(total)) % MOD
    print(result)

solve()