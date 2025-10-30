class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # prefix_or[i] = OR of nums[0..i-1] 
        prefix_or = [0] * n
        for i in range(1, n):
            prefix_or[i] = prefix_or[i-1] | nums[i-1]
        
        # suffix_or[i] = OR of nums[i+1..n-1]
        suffix_or = [0] * n
        for i in range(n-2, -1, -1):
            suffix_or[i] = suffix_or[i+1] | nums[i+1]
        
        max_or = 0
        multiplier = 2 ** k
        
        for i in range(n):
            # Apply all k operations to nums[i]
            current_or = prefix_or[i] | (nums[i] * multiplier) | suffix_or[i]
            max_or = max(max_or, current_or)
        
        return max_or