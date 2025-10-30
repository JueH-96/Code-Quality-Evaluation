class Solution:
    def maxSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        # Function to find the maximum digit in a number
        def max_digit(num):
            return max(int(d) for d in str(num))
        
        # Dictionary to store the maximum sum for each maximum digit
        max_digit_map = defaultdict(int)
        
        max_sum = -1
        
        for num in nums:
            max_d = max_digit(num)
            if max_d in max_digit_map:
                # Calculate potential new max sum
                potential_sum = max_digit_map[max_d] + num
                # Update the global max sum if the potential sum is greater
                max_sum = max(max_sum, potential_sum)
                # Update the maximum number for this max digit
                max_digit_map[max_d] = max(max_digit_map[max_d], num)
            else:
                # If this is the first number with this max digit, just store it
                max_digit_map[max_d] = num
        
        return max_sum