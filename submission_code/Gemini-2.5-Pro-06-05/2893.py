import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        """
        Calculates the maximum score achievable by visiting a subsequence of positions.

        This problem is solved using dynamic programming. The state we need to maintain
        as we iterate through the array is the maximum score for a path ending in an
        even number and the maximum score for a path ending in an odd number.
        """
        
        # max_even_score: max score for any valid path ending in an even number.
        # max_odd_score: max score for any valid path ending in an odd number.
        max_even_score = -math.inf
        max_odd_score = -math.inf

        # We must start at index 0. Initialize scores based on the first element.
        if nums[0] % 2 == 0:
            max_even_score = nums[0]
        else:
            max_odd_score = nums[0]

        # Iterate through the rest of the array.
        for i in range(1, len(nums)):
            num = nums[i]
            
            if num % 2 == 0:  # Current number is even
                # A path ending at `i` can extend a previous even-ending path
                # or a previous odd-ending path.
                # Score if extending from even: max_even_score + num
                # Score if extending from odd: max_odd_score + num - x
                
                # The best score for a new path ending at `i`
                new_path_score = num + max(max_even_score, max_odd_score - x)
                
                # Update max_even_score. Taking the max with its current value
                # handles the case where it's better to not include `num` (i.e., skip it).
                max_even_score = max(max_even_score, new_path_score)
                
            else:  # Current number is odd
                # Similar logic as the even case.
                # Score if extending from odd: max_odd_score + num
                # Score if extending from even: max_even_score + num - x

                new_path_score = num + max(max_odd_score, max_even_score - x)
                
                max_odd_score = max(max_odd_score, new_path_score)

        # The final answer is the maximum of the two possible path-ending parities.
        return int(max(max_even_score, max_odd_score))