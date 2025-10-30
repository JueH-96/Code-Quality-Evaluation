from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        """
        Transforms an array by replacing even numbers with 0, odd numbers with 1,
        and then sorting the result.
        
        Args:
            nums: A list of integers.
            
        Returns:
            The transformed and sorted list.
        """
        
        # Step 1 & 2: Replace evens with 0 and odds with 1.
        # The expression `num % 2` naturally maps even numbers to 0 and odd numbers to 1.
        # A list comprehension is used to efficiently create the transformed list.
        result = [num % 2 for num in nums]
        
        # Step 3: Sort the modified array in non-decreasing order.
        # The .sort() method sorts the list in-place.
        result.sort()
        
        return result