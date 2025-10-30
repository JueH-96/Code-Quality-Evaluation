class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        mod = 10**9 + 7
        T = m * n
        # If fewer than 2 pieces or more pieces than cells, distance sum is 0
        if k < 2 or k > T:
            return 0
        
        # Precompute factorials and inverse factorials up to T
        fac = [1] * (T + 1)
        invfac = [1] * (T + 1)
        for i in range(1, T + 1):
            fac[i] = fac[i-1] * i % mod
        invfac[T] = pow(fac[T], mod - 2, mod)
        for i in range(T, 0, -1):
            invfac[i-1] = invfac[i] * i % mod
        
        def comb(n, r):
            if r < 0 or r > n:
                return 0
            return fac[n] * invfac[r] % mod * invfac[n-r] % mod
        
        # Number of ways to choose the remaining k-2 pieces once we fix a pair
        ways_with_pair = comb(T-2, k-2)
        
        # Sum of row‑differences over all unordered row‑pairs:
        #   sum_{1 <= i < j <= m} (j - i) = (m-1)*m*(m+1)/6
        inv6 = pow(6, mod-2, mod)
        sum_row_diff = (m-1) * m % mod * (m+1) % mod * inv6 % mod
        
        # Sum of col‑differences over all unordered col‑pairs:
        #   sum_{1 <= i < j <= n} (j - i) = (n-1)*n*(n+1)/6
        sum_col_diff = (n-1) * n % mod * (n+1) % mod * inv6 % mod
        
        # Total Manhattan sum over all unordered cell‑pairs:
        #   S = n^2 * sum_row_diff + m^2 * sum_col_diff
        S = (n*n % mod) * sum_row_diff % mod
        S = (S + (m*m % mod) * sum_col_diff % mod) % mod
        
        # Multiply by the number of ways to place the other k-2 pieces
        return ways_with_pair * S % mod