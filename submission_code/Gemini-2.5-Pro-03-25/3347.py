import collections # This import is not strictly necessary for this solution, but potentially useful in other contexts. Can be removed if not used. Let's remove it.
from typing import List

class Solution:
    """
    Implements the logic to distribute elements of nums into two arrays arr1 and arr2
    based on the specified rules and returns the concatenated result.
    The input `nums` is assumed to be a 0-indexed Python list, despite the problem description
    mentioning 1-based indexing for clarity.
    """
    def resultArray(self, nums: List[int]) -> List[int]:
        """
        Distributes elements of nums into arr1 and arr2 according to the rules and 
        returns their concatenation.

        Args:
            nums: A list of distinct integers. The length n is guaranteed to be >= 3.

        Returns:
            A list representing the concatenation of arr1 and arr2 after distributing 
            all elements from nums according to the specified rules.
        """
        
        # Get the length of the input array.
        n = len(nums)

        # The problem statement guarantees n >= 3, so nums has at least 3 elements.
        # This ensures nums[0] and nums[1] are always valid indices to access.

        # Initialize arr1 with the first element of nums (corresponds to the 1st operation).
        # The problem states: "In the first operation, append nums[1] to arr1."
        # Assuming 0-based indexing for the input list `nums`, this corresponds to `nums[0]`.
        arr1 = [nums[0]]
        
        # Initialize arr2 with the second element of nums (corresponds to the 2nd operation).
        # The problem states: "In the second operation, append nums[2] to arr2."
        # Assuming 0-based indexing for `nums`, this corresponds to `nums[1]`.
        arr2 = [nums[1]]

        # Process the remaining elements starting from the third element (index 2).
        # This corresponds to operations i = 3 to n as described in the problem.
        # The loop iterates using 0-based Python indices i = 2, 3, ..., n-1, which correctly
        # accesses the 3rd, 4th, ..., nth elements of the original list `nums`.
        for i in range(2, n):
            # Get the current element from nums to be distributed. This corresponds to nums[i+1] in 1-based indexing.
            current_num = nums[i]

            # Compare the last elements of arr1 and arr2 to decide where to append the current element.
            # Accessing arr1[-1] and arr2[-1] is safe because both arrays are initialized with one element
            # and elements are only appended, ensuring they are never empty after initialization.
            if arr1[-1] > arr2[-1]:
                # If the last element of arr1 is greater than the last element of arr2, append to arr1.
                arr1.append(current_num)
            else:
                # Otherwise (last element of arr1 <= last element of arr2), append to arr2.
                arr2.append(current_num)

        # Concatenate arr1 and arr2 to form the final result array.
        # The '+' operator for Python lists creates a new list containing elements of arr1 followed by elements of arr2.
        result = arr1 + arr2

        # Return the concatenated array.
        return result