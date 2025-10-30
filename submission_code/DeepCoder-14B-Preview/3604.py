MOD = 10**9 + 7
max_n = 1000

# Precompute Stirling numbers of the second kind
stirling = [[0] * (max_n + 1) for _ in range(max_n + 1)]
stirling[0][0] = 1
for i in range(1, max_n + 1):
    for j in range(1, i + 1):
        stirling[i][j] = (stirling[i-1][j-1] + j * stirling[i-1][j]) % MOD

# Precompute factorials modulo MOD
fact = [1] * (max_n + 1)
for i in range(1, max_n + 1):
    fact[i] = fact[i-1] * i % MOD

# Precompute inverse factorials modulo MOD
inv_fact = [1] * (max_n + 1)
inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
for i in range(max_n - 1, -1, -1):
    inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        total = 0
        max_k = min(x, n)
        for k in range(1, max_k + 1):
            # Calculate combination C(x, k)
            if x < k or k < 0:
                c = 0
            else:
                c = fact[x] * inv_fact[k] % MOD
                c = c * inv_fact[x - k] % MOD
            
            # Calculate k! * S(n, k)
            if n < k:
                s = 0
            else:
                s = fact[k] * stirling[n][k] % MOD
            
            # Calculate the term for this k
            term = c * s % MOD
            term = term * pow(y, k, MOD) % MOD
            total = (total + term) % MOD
        
        return total