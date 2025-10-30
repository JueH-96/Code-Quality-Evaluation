class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        from functools import lru_cache
        
        # Helper function to compute maximum XOR subarray in a given range
        def max_xor_subarray(l, r):
            max_xor = 0
            for start in range(l, r + 1):
                current_xor = 0
                for end in range(start, r + 1):
                    current_xor ^= nums[end]
                    max_xor = max(max_xor, current_xor)
            return max_xor
        
        # Precompute results for all queries
        results = []
        for l, r in queries:
            results.append(max_xor_subarray(l, r))
        
        return results