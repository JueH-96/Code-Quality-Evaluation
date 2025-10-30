class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        mod = 10**9 + 7
        # If k > n-1 it's impossible to have that many equal adjacent pairs
        if k > n - 1:
            return 0
        
        # We use the formula:
        #   answer = m * C(n-1, k) * (m-1)^(n-1-k)  (mod 1e9+7)
        # Precompute factorials and inverse factorials up to n-1
        # so we can compute C(n-1, k) quickly.
        
        # factorial[i] = i! % mod
        factorial = [1] * n
        for i in range(1, n):
            factorial[i] = factorial[i-1] * i % mod
        
        # inv_factorial[i] = (i!)^-1 % mod
        inv_factorial = [1] * n
        inv_factorial[n-1] = pow(factorial[n-1], mod-2, mod)
        for i in range(n-1, 0, -1):
            inv_factorial[i-1] = inv_factorial[i] * i % mod
        
        # Compute C(n-1, k)
        comb = factorial[n-1]
        comb = comb * inv_factorial[k] % mod
        comb = comb * inv_factorial[n-1-k] % mod
        
        # Compute the power term (m-1)^(n-1-k)
        pow_term = pow(m-1, n-1-k, mod)
        
        # Final answer
        return m * comb % mod * pow_term % mod