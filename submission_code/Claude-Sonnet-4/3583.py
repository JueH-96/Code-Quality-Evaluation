class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        from math import gcd
        from collections import Counter
        
        # Count frequency of each number
        counter = Counter(nums)
        max_val = max(nums)
        
        # Count how many numbers are divisible by each value
        divisible_count = [0] * (max_val + 1)
        for i in range(1, max_val + 1):
            for j in range(i, max_val + 1, i):
                if j in counter:
                    divisible_count[i] += counter[j]
        
        # Count pairs with GCD exactly equal to g using inclusion-exclusion
        gcd_pair_count = [0] * (max_val + 1)
        
        # Start from max_val and work backwards
        for g in range(max_val, 0, -1):
            # Count of numbers divisible by g
            cnt = divisible_count[g]
            # Total pairs that have GCD divisible by g
            pairs_divisible_by_g = cnt * (cnt - 1) // 2
            
            # Subtract pairs that have GCD > g (already calculated)
            gcd_pair_count[g] = pairs_divisible_by_g
            for multiple in range(2 * g, max_val + 1, g):
                gcd_pair_count[g] -= gcd_pair_count[multiple]
        
        # Build the sorted gcd array by expanding counts
        gcd_pairs = []
        for g in range(1, max_val + 1):
            gcd_pairs.extend([g] * gcd_pair_count[g])
        
        # Answer queries
        result = []
        for query in queries:
            result.append(gcd_pairs[query])
        
        return result