class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # Initialize dp arrays for even and odd parities
        dp_even = -float('inf')
        dp_odd = -float('inf')
        
        # Initialize based on the first element
        if nums[0] % 2 == 0:
            dp_even = nums[0]
        else:
            dp_odd = nums[0]
        
        for i in range(1, n):
            current_num = nums[i]
            if current_num % 2 == 0:
                # Current number is even
                # Option 1: Continue from previous even
                option1 = dp_even + current_num
                # Option 2: Switch from odd, lose x
                option2 = dp_odd + current_num - x
                dp_even = max(option1, option2)
            else:
                # Current number is odd
                # Option 1: Continue from previous odd
                option1 = dp_odd + current_num
                # Option 2: Switch from even, lose x
                option2 = dp_even + current_num - x
                dp_odd = max(option1, option2)
        
        return max(dp_even, dp_odd)