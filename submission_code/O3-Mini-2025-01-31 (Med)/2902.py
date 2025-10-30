from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Dictionary to map the maximum digit in the number to a list of numbers having that max digit.
        groups = {}
        
        # Helper function to compute maximum digit of a number.
        def max_digit(num: int) -> int:
            return max(int(d) for d in str(num))
        
        # Populate the groups dictionary.
        for num in nums:
            key = max_digit(num)
            if key in groups:
                groups[key].append(num)
            else:
                groups[key] = [num]
        
        max_sum = -1
        # For each group, if there are at least two numbers, compute the sum of the two largest numbers.
        for key, numbers in groups.items():
            if len(numbers) >= 2:
                # sort the list in descending order to get the two largest numbers
                numbers.sort(reverse=True)
                current_sum = numbers[0] + numbers[1]
                max_sum = max(max_sum, current_sum)
        
        return max_sum