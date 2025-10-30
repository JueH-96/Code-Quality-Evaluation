class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Precompute prefix OR
        prefix_or = [0] * n
        prefix_or[0] = 0
        for i in range(1, n):
            prefix_or[i] = prefix_or[i-1] | nums[i-1]
        
        # Precompute suffix OR
        suffix_or = [0] * n
        suffix_or[n-1] = 0
        for i in range(n-2, -1, -1):
            suffix_or[i] = suffix_or[i+1] | nums[i+1]
        
        max_or = 0
        multiplier = 1 << k  # 2^k
        
        # Try applying all k operations to each position
        for i in range(n):
            current_or = prefix_or[i] | (nums[i] * multiplier) | suffix_or[i]
            max_or = max(max_or, current_or)
        
        return max_or