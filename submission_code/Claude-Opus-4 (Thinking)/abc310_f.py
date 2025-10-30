# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

MOD = 998244353

# dp[mask] = number of ways to achieve the sums represented by mask
# bit i in mask indicates whether sum i is achievable
dp = {1: 1}  # Initially only sum 0 is achievable (empty subset)

for i in range(n):
    new_dp = {}
    
    for mask, count in dp.items():
        # For die values 1 to min(a[i], 10)
        for v in range(1, min(a[i] + 1, 11)):
            new_mask = mask
            # Add v to all previously achievable sums
            for s in range(11):
                if mask & (1 << s) and s + v <= 10:
                    new_mask |= (1 << (s + v))
            
            new_dp[new_mask] = new_dp.get(new_mask, 0) + count
            new_dp[new_mask] %= MOD
        
        # For die values > 10 (if any)
        if a[i] > 10:
            # These values don't help achieve any sum <= 10
            new_dp[mask] = new_dp.get(mask, 0) + count * (a[i] - 10)
            new_dp[mask] %= MOD
    
    dp = new_dp

# Count configurations where sum 10 is achievable
favorable = 0
for mask, count in dp.items():
    if mask & (1 << 10):  # Check if bit 10 is set
        favorable = (favorable + count) % MOD

# Total number of configurations
total = 1
for x in a:
    total = (total * x) % MOD

# Compute answer using modular inverse
answer = (favorable * pow(total, MOD - 2, MOD)) % MOD
print(answer)