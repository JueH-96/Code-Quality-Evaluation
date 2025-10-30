class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # Initialize max_even and max_odd based on the parity of nums[0]
        if nums[0] % 2 == 0:
            max_even = nums[0]
            max_odd = float('-inf')
        else:
            max_odd = nums[0]
            max_even = float('-inf')
        
        for i in range(1, len(nums)):
            num = nums[i]
            if num % 2 == 0:
                # Update max_even considering transitions from even and odd
                new_max_even = max(max_even + num, max_odd + num - x)
                new_max_odd = max_odd  # max_odd remains unchanged
            else:
                # Update max_odd considering transitions from odd and even
                new_max_odd = max(max_odd + num, max_even + num - x)
                new_max_even = max_even  # max_even remains unchanged
            # Update the max_even and max_odd for the next iteration
            max_even, max_odd = new_max_even, new_max_odd
        
        # Return the maximum of max_even and max_odd
        return int(max(max_even, max_odd))