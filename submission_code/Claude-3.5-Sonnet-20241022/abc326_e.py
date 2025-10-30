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

N = int(input())
A = list(map(int, input().split()))
MOD = 998244353

# dp[i] = expected value when x = i
dp = [0] * (N + 1)

# Solve from N-1 to 0
for i in range(N-1, -1, -1):
    # Count valid transitions
    valid_transitions = 0
    value_sum = 0
    
    # For each possible roll
    for j in range(N):
        roll = j + 1
        if roll > i:
            valid_transitions += 1
            value_sum = (value_sum + A[j] + dp[roll]) % MOD
    
    if valid_transitions == 0:
        dp[i] = 0
    else:
        # Calculate probability denominator inverse
        prob_denom = mod_inverse(N, MOD)
        dp[i] = (value_sum * prob_denom) % MOD

print(dp[0])