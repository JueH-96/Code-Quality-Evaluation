class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        import sys
        
        n = len(nums)
        
        # Precompute prefix OR
        prefix = [0] * n
        prefix[0] = nums[0]
        for i in range(1, n):
            prefix[i] = prefix[i-1] | nums[i]
        
        # Precompute suffix OR
        suffix = [0] * n
        suffix[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffix[i] = suffix[i+1] | nums[i]
        
        # Function to get prefix OR up to index-1 (or 0 if index is 0)
        def get_prefix_or(index: int) -> int:
            if index < 0:
                return 0
            return prefix[index]
        
        # Function to get suffix OR from index+1 (or 0 if index is last)
        def get_suffix_or(index: int) -> int:
            if index >= n:
                return 0
            return suffix[index]
        
        # Try applying all k shifts to each element and compute the OR
        best = 0
        for i in range(n):
            left_or = get_prefix_or(i-1)
            right_or = get_suffix_or(i+1)
            shifted = nums[i] << k
            candidate = left_or | shifted | right_or
            best = max(best, candidate)
        
        return best