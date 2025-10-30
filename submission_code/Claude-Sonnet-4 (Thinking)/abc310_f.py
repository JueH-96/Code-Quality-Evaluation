n = int(input())
A = list(map(int, input().split()))

MOD = 998244353

# dp[S] = number of outcomes such that the set of achievable sums is S
# S is represented as a bitmask
dp = [0] * (1 << 11)
dp[1] = 1  # Initially, only sum 0 is achievable

for i in range(n):
    new_dp = [0] * (1 << 11)
    for mask in range(1 << 11):
        if dp[mask] == 0:
            continue
        
        # Handle results in [1, min(A[i], 10)]
        for result in range(1, min(A[i], 10) + 1):
            new_mask = mask  # We can choose not to include the current die
            for s in range(11):
                if (mask >> s) & 1:  # s is achievable
                    if s + result <= 10:
                        new_mask |= (1 << (s + result))  # We can choose to include the current die
            new_dp[new_mask] = (new_dp[new_mask] + dp[mask]) % MOD
        
        # Handle results in [min(A[i], 10) + 1, A[i]]
        if A[i] > 10:
            new_dp[mask] = (new_dp[mask] + dp[mask] * (A[i] - 10)) % MOD
    
    dp = new_dp

# Count outcomes where sum 10 is achievable
satisfying_outcomes = 0
for mask in range(1 << 11):
    if (mask >> 10) & 1:  # sum 10 is achievable
        satisfying_outcomes = (satisfying_outcomes + dp[mask]) % MOD

total_outcomes = 1
for a in A:
    total_outcomes = (total_outcomes * a) % MOD

# Compute probability modulo 998244353
inv_total = pow(total_outcomes, MOD - 2, MOD)
result = (satisfying_outcomes * inv_total) % MOD
print(result)