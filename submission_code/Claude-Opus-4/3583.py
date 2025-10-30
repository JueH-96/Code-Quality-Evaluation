class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        from math import gcd
        from collections import Counter
        
        # Count frequency of each number in nums
        freq = Counter(nums)
        max_val = max(nums)
        
        # Count pairs for each GCD value
        gcd_count = [0] * (max_val + 1)
        
        # For each possible GCD value g
        for g in range(1, max_val + 1):
            # Count numbers that are multiples of g
            multiples = []
            for num in freq:
                if num % g == 0:
                    multiples.append(num)
            
            if len(multiples) < 2 and (len(multiples) != 1 or freq[multiples[0]] < 2):
                continue
            
            # Count pairs where both numbers are multiples of g
            pairs_with_g_as_divisor = 0
            for i in range(len(multiples)):
                for j in range(i, len(multiples)):
                    if i == j:
                        # Same number, choose 2 from freq[multiples[i]]
                        pairs_with_g_as_divisor += freq[multiples[i]] * (freq[multiples[i]] - 1) // 2
                    else:
                        # Different numbers
                        pairs_with_g_as_divisor += freq[multiples[i]] * freq[multiples[j]]
            
            # Use inclusion-exclusion to get pairs with GCD exactly g
            gcd_count[g] = pairs_with_g_as_divisor
            
            # Subtract pairs where GCD is a multiple of g (already counted)
            for multiple in range(2 * g, max_val + 1, g):
                gcd_count[g] -= gcd_count[multiple]
        
        # Build the sorted gcdPairs array
        gcdPairs = []
        for g in range(1, max_val + 1):
            if gcd_count[g] > 0:
                gcdPairs.extend([g] * gcd_count[g])
        
        # Answer queries
        return [gcdPairs[q] for q in queries]