class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        from math import gcd
        from functools import reduce
        
        MOD = 10**9 + 7
        n = len(nums)
        
        # Helper function to compute GCD of elements at given indices
        def compute_gcd(indices):
            if not indices:
                return 0
            arr = [nums[i] for i in indices]
            return reduce(gcd, arr)
        
        # For each possible GCD, store all subsequences (as bitmasks) that have that GCD
        gcd_to_masks = {}
        
        # Generate all possible non-empty subsequences
        for mask in range(1, 1 << n):
            indices = []
            for i in range(n):
                if mask & (1 << i):
                    indices.append(i)
            
            subseq_gcd = compute_gcd(indices)
            if subseq_gcd not in gcd_to_masks:
                gcd_to_masks[subseq_gcd] = []
            gcd_to_masks[subseq_gcd].append(mask)
        
        result = 0
        
        # For each GCD value, count pairs of disjoint subsequences
        for g, masks in gcd_to_masks.items():
            # Count pairs of disjoint masks
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    # Check if masks are disjoint (no common set bits)
                    if masks[i] & masks[j] == 0:
                        result = (result + 1) % MOD
        
        return result