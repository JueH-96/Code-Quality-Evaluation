MOD = 10**9 + 7

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        res = 0
        
        # Precompute the powers of 2 up to n, modulo MOD
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = (pow2[i-1] * 2) % MOD
        
        # Precompute the prefix sums of combinations: sum_{j=0 to m} C(m, j) = 2^m, but we need sum up to min(m, k-1)
        # So for each element nums[i], as min: the count is sum_{j=0}^{min(r, k-1)} C(r, j), where r is the number of elements to the right (n - i - 1)
        # Similarly for max: sum_{j=0}^{min(l, k-1)} C(l, j), l = i
        
        # We can compute for each i, the sum_{j=0 to m} C(m, j) for m up to n-1, and j up to k-1
        # But for large m (1e5) and k up to 70, we need an efficient way
        
        # Precompute factorial, inverse factorial up to n for mod calculations
        fact = [1] * (n + 1)
        inv_fact = [1] * (n + 1)
        
        for i in range(1, n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact[n] = pow(fact[n], MOD-2, MOD)
        for i in range(n-1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        def comb(m, j):
            if m < 0 or j < 0 or j > m:
                return 0
            return fact[m] * inv_fact[j] % MOD * inv_fact[m - j] % MOD
        
        # Precompute for each possible m, the sum_{j=0}^{min(m, k-1)} C(m, j)
        sum_comb = [0] * (n + 2)
        for m in range(0, n + 1):
            max_j = min(m, k - 1)
            s = 0
            for j in range(0, max_j + 1):
                s = (s + comb(m, j)) % MOD
            sum_comb[m] = s
        
        for i in range(n):
            x = nums[i]
            # Number of times x is min: sum_{s=1 to k} (number of subsets where x is min and size is s)
            # For x to be min, all other elements in the subset must be >=x. The other elements can be any of the (n - i - 1) elements to the right.
            # The number of such subsets of size s is C(r, s-1), where r = n - i - 1.
            # Total for all s up to k is sum_{s=1 to k} C(r, s-1) = sum_{j=0 to k-1} C(r, j)
            r = n - i - 1
            cnt_min = sum_comb[r]
            
            # Number of times x is max: similarly, elements must be <=x. So elements are from 0..i-1 (l = i elements before)
            # For subsets of size s: sum_{s=1 to k} C(l, s-1) = sum_{j=0 to k-1} C(l, j)
            l = i
            cnt_max = sum_comb[l]
            
            total = (cnt_min + cnt_max) % MOD
            res = (res + x * total) % MOD
        
        return res