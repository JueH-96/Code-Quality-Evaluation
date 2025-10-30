class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        if not nums:
            return 0
        
        # Initialize previous even and odd states
        prev_even = nums[0] if nums[0] % 2 == 0 else float('-inf')
        prev_odd = nums[0] if nums[0] % 2 != 0 else float('-inf')
        
        for num in nums[1:]:
            if num % 2 == 0:
                # Calculate options for current even state
                option1 = prev_even + num
                option2 = prev_odd + num - x if prev_odd != float('-inf') else float('-inf')
                current_even = max(option1, option2)
                current_odd = float('-inf')
            else:
                # Calculate options for current odd state
                option1 = prev_odd + num
                option2 = prev_even + num - x if prev_even != float('-inf') else float('-inf')
                current_odd = max(option1, option2)
                current_even = float('-inf')
            
            # Update previous values for the next iteration
            prev_even, prev_odd = current_even, current_odd
        
        # The result is the maximum of the two possible states, ensuring it's not negative infinity
        return max(prev_even, prev_odd) if max(prev_even, prev_odd) != float('-inf') else 0