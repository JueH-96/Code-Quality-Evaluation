class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # Initialize two variables to keep track of the maximum score ending with an even and an odd number
        max_even = max_odd = float('-inf')
        
        # Set the initial score based on the parity of the first element
        if nums[0] % 2 == 0:
            max_even = nums[0]
        else:
            max_odd = nums[0]
        
        # Iterate through the array starting from the second element
        for num in nums[1:]:
            if num % 2 == 0:
                # If the current number is even, update max_even
                max_even = num + max(max_even, max_odd - x)
            else:
                # If the current number is odd, update max_odd
                max_odd = num + max(max_odd, max_even - x)
        
        # The result will be the maximum score ending with either an even or an odd number
        return max(max_even, max_odd)