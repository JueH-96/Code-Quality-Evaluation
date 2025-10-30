import collections
from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        """
        Finds the maximum sum of a pair of numbers from nums such that the 
        maximum digit in both numbers are equal.

        Args:
            nums: A 0-indexed integer array.

        Returns:
            The maximum sum, or -1 if no such pair exists.
        """

        def get_max_digit(n: int) -> int:
            """Helper function to find the maximum digit in a number."""
            max_d = 0
            # Handle the case of 0 separately if constraints allowed 0
            # if n == 0:
            #     return 0 
            # Since 1 <= nums[i], we don't need to worry about n=0
            
            s = str(n)
            for char in s:
                max_d = max(max_d, int(char))
            
            # Alternative using integer arithmetic:
            # while n > 0:
            #     digit = n % 10
            #     max_d = max(max_d, digit)
            #     n //= 10
                
            return max_d

        # Dictionary to store the largest number encountered so far for each max digit.
        # We actually need the two largest numbers for each max digit to form a pair.
        # We can store a list of numbers for each max digit and find the top two later,
        # or maintain the top two as we iterate. Let's maintain the top two.
        
        # Stores {max_digit: (largest_num, second_largest_num)}
        max_nums_by_digit = collections.defaultdict(lambda: (-1, -1)) 
        max_overall_sum = -1

        for num in nums:
            max_d = get_max_digit(num)
            
            largest, second_largest = max_nums_by_digit[max_d]

            # Update the largest and second largest for this max_d
            if num >= largest:
                # Current num is the new largest, old largest becomes second largest
                max_nums_by_digit[max_d] = (num, largest)
                # Update second_largest variable for potential sum calculation below
                second_largest = largest 
            elif num > second_largest:
                 # Current num is smaller than largest, but larger than second largest
                max_nums_by_digit[max_d] = (largest, num)
                 # Update second_largest variable for potential sum calculation below
                second_largest = num
            
            # If we have found at least two numbers for this max_d,
            # check if their sum is the new maximum overall sum.
            # We check `second_largest > -1` (or > 0 since nums[i] >= 1) 
            # to ensure we have found two distinct numbers (or potentially the same number twice if duplicates allowed).
            # The logic above correctly handles duplicates - if num == largest, it gets stored as largest, and the old largest becomes second_largest.
            if second_largest != -1: 
                 # Re-fetch the potentially updated largest number
                largest, _ = max_nums_by_digit[max_d] 
                current_sum = largest + second_largest
                max_overall_sum = max(max_overall_sum, current_sum)

        return max_overall_sum