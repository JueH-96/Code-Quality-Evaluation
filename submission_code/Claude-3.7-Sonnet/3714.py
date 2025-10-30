class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        from collections import Counter
        from math import comb
        
        # Count frequencies of each value
        count = Counter(nums)
        sorted_values = sorted(count.keys())
        
        # Precompute counts of elements less than and greater than each value
        less_count = {}
        greater_count = {}
        
        total_less = 0
        for v in sorted_values:
            less_count[v] = total_less
            total_less += count[v]
        
        total_greater = 0
        for v in reversed(sorted_values):
            greater_count[v] = total_greater
            total_greater += count[v]
        
        # Helper function to calculate sum of combinations
        def sum_combinations(n, k):
            """Computes the sum of C(n, r) for r from 0 to k."""
            result = 0
            for r in range(min(k + 1, n + 1)):
                result = (result + comb(n, r)) % MOD
            return result
        
        result = 0
        for v in sorted_values:
            # Calculate ways where v is the minimum
            min_ways = 0
            for r in range(1, min(k, count[v]) + 1):
                # Choose r occurrences of v and up to (k-r) elements greater than v
                ways = (comb(count[v], r) * sum_combinations(greater_count[v], k - r)) % MOD
                min_ways = (min_ways + ways) % MOD
            
            # Calculate ways where v is the maximum
            max_ways = 0
            for r in range(1, min(k, count[v]) + 1):
                # Choose r occurrences of v and up to (k-r) elements less than v
                ways = (comb(count[v], r) * sum_combinations(less_count[v], k - r)) % MOD
                max_ways = (max_ways + ways) % MOD
            
            # Add contribution of value v to the result
            result = (result + v * (min_ways + max_ways)) % MOD
        
        return result