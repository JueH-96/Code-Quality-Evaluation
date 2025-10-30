from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Function to find the maximum digit in a number
        def max_digit(num):
            return max(int(d) for d in str(num))
        
        # Group numbers by their maximum digits
        groups = {}
        for num in nums:
            md = max_digit(num)
            if md in groups:
                groups[md].append(num)
            else:
                groups[md] = [num]
        
        # Initialize maximum sum to -1
        max_sum = -1
        
        # Iterate through each group to find the maximum sum of pairs
        for group in groups.values():
            if len(group) >= 2:
                # Sort the group in descending order
                group_sorted = sorted(group, reverse=True)
                # Sum the top two numbers
                current_sum = group_sorted[0] + group_sorted[1]
                # Update max_sum if current_sum is larger
                if current_sum > max_sum:
                    max_sum = current_sum
        
        # Return the maximum sum found, or -1 if no pair exists
        return max_sum