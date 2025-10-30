class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_or = [0] * n
        suffix_or = [0] * n
        
        # Calculate prefix OR
        for i in range(1, n):
            prefix_or[i] = prefix_or[i - 1] | nums[i - 1]
        
        # Calculate suffix OR
        for i in range(n - 2, -1, -1):
            suffix_or[i] = suffix_or[i + 1] | nums[i + 1]
        
        max_or = 0
        
        # Try multiplying each element by 2^k and calculate the OR
        for i in range(n):
            current_or = prefix_or[i] | (nums[i] << k) | suffix_or[i]
            max_or = max(max_or, current_or)
        
        return max_or