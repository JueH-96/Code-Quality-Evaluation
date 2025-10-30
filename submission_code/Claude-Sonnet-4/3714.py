class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Precompute factorials and inverse factorials for combinations
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        def pow_mod(base, exp, mod):
            result = 1
            while exp > 0:
                if exp % 2 == 1:
                    result = result * base % mod
                base = base * base % mod
                exp //= 2
            return result
        
        inv_fact = [1] * (n + 1)
        inv_fact[n] = pow_mod(fact[n], MOD - 2, MOD)
        for i in range(n - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        def comb(n, r):
            if r < 0 or r > n:
                return 0
            return fact[n] * inv_fact[r] % MOD * inv_fact[n - r] % MOD
        
        result = 0
        
        # For each element, calculate its contribution as minimum and maximum
        for i in range(n):
            # Contribution as minimum
            # Can choose from positions i to n-1 (n-i elements available)
            # Need to form subsequences of size 1 to k
            min_contrib = 0
            for size in range(1, min(k + 1, n - i + 1)):
                # Choose size-1 elements from the n-i-1 elements after position i
                count = comb(n - i - 1, size - 1)
                min_contrib = (min_contrib + count) % MOD
            
            # Contribution as maximum  
            # Can choose from positions 0 to i (i+1 elements available)
            # Need to form subsequences of size 1 to k
            max_contrib = 0
            for size in range(1, min(k + 1, i + 2)):
                # Choose size-1 elements from the i elements before position i
                count = comb(i, size - 1)
                max_contrib = (max_contrib + count) % MOD
            
            # Add this element's total contribution
            element_contrib = (nums[i] * (min_contrib + max_contrib)) % MOD
            result = (result + element_contrib) % MOD
        
        return result