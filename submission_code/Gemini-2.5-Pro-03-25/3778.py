import collections
from typing import List

class Solution:
    """
    Transforms an array of integers according to specified rules:
    1. Replaces even numbers with 0.
    2. Replaces odd numbers with 1.
    3. Sorts the modified array non-decreasingly.
    """
    def transformArray(self, nums: List[int]) -> List[int]:
        """
        Performs the transformation and sorting operations on the input list.

        Args:
            nums: The input list of integers.

        Returns:
            The transformed and sorted list of integers (containing only 0s and 1s).
        """
        
        # Step 1 & 2: Replace even numbers with 0 and odd numbers with 1.
        # We can do this efficiently using a list comprehension.
        # For each number 'x' in 'nums', check if it's even (x % 2 == 0).
        # If it is, the new value is 0. Otherwise (it's odd), the new value is 1.
        transformed_nums = [0 if x % 2 == 0 else 1 for x in nums]
        
        # Step 3: Sort the modified array in non-decreasing order.
        # The built-in sort() method sorts the list in place non-decreasingly.
        transformed_nums.sort() 
        # Alternatively, we could use sorted() which returns a new sorted list:
        # sorted_nums = sorted(transformed_nums)
        # return sorted_nums
        
        # Return the resulting array.
        return transformed_nums