class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # Initialize based on starting position
        if nums[0] % 2 == 0:
            max_even = nums[0]
            max_odd = float('-inf')
        else:
            max_even = float('-inf')
            max_odd = nums[0]
        
        # Process each position after the start
        for i in range(1, len(nums)):
            if nums[i] % 2 == 0:  # Current number is even
                # Can come from best even (no penalty) or best odd (with penalty)
                max_even = max(max_even + nums[i], max_odd + nums[i] - x)
            else:  # Current number is odd
                # Can come from best odd (no penalty) or best even (with penalty)
                max_odd = max(max_odd + nums[i], max_even + nums[i] - x)
        
        return max(max_even, max_odd)