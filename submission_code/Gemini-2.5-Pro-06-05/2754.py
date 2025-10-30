import math
from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        """
        Calculates the maximum strength of a non-empty group of students.
        The strength is the product of the scores of the students in the group.
        """
        
        # Handle the base case of a single element array.
        if len(nums) == 1:
            return nums[0]

        # Partition the numbers into positive, negative, and check for zeros.
        positives = [x for x in nums if x > 0]
        negatives = [x for x in nums if x < 0]
        has_zero = any(x == 0 for x in nums)

        # Handle edge cases where the max strength might not be positive.
        # This occurs if we have no positive numbers and at most one negative number.
        if not positives and len(negatives) <= 1:
            # If a zero exists, or if the list is all zeros, the max strength is 0
            # (from the group [0]), which is better than a negative strength.
            if has_zero or not negatives:
                return 0
            # Otherwise, the list contains a single negative number and nothing else.
            # The only possible non-empty group is the number itself.
            else: # len(negatives) == 1 and not has_zero
                return negatives[0]

        # In all other scenarios, we can construct a positive product.
        
        # Start with the product of all positive numbers.
        # math.prod on an empty list returns 1, which is a perfect identity.
        strength = 1
        if positives:
            strength = math.prod(positives)

        # To keep the strength positive, we must multiply by an even number of negatives.
        
        # If the count of negative numbers is even, we can include all of them.
        if len(negatives) % 2 == 0:
            strength *= math.prod(negatives)
        # If the count is odd, we must exclude one. To maximize the product,
        # we exclude the negative number with the smallest absolute value 
        # (i.e., the one closest to zero, which is the largest value).
        else:
            negatives.sort() # Sorts from most negative to least (e.g., [-9, -5, -1])
            # The product of all but the last element will be positive and maximized.
            # math.prod of an empty list (if len(negatives)==1) is 1, which is correct.
            strength *= math.prod(negatives[:-1])
        
        return strength