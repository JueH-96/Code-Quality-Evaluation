from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Dictionary to group numbers by their maximum digit
        digit_groups = defaultdict(list)
        
        # Function to find the maximum digit in a number
        def find_max_digit(n):
            max_digit = 0
            while n > 0:
                digit = n % 10
                if digit > max_digit:
                    max_digit = digit
                n = n // 10
            return max_digit
        
        # Group numbers by their maximum digit
        for num in nums:
            max_digit = find_max_digit(num)
            digit_groups[max_digit].append(num)
        
        # Initialize the maximum sum to -1
        max_sum = -1
        
        # For each group with at least two numbers
        for group in digit_groups.values():
            if len(group) >= 2:
                # Find the two largest numbers in the group
                max1 = max2 = float('-inf')
                for num in group:
                    if num > max1:
                        max2 = max1
                        max1 = num
                    elif num > max2:
                        max2 = num
                # Calculate their sum and update the maximum sum if necessary
                current_sum = max1 + max2
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum