from typing import List

class Solution:
  def resultArray(self, nums: List[int]) -> List[int]:
    # Initialize arr1 and arr2
    arr1 = []
    arr2 = []

    # The problem statement describes nums as 1-indexed.
    # The input `nums: List[int]` is a 0-indexed Python list.
    # So, problem's conceptual nums[k] corresponds to Python's nums[k-1].

    # Constraint: 3 <= n <= 50, where n is the length of nums.
    # This ensures that nums has at least 3 elements, so nums[0] and nums[1]
    # (corresponding to problem's nums[1] and nums[2]) are always accessible.

    # First operation: "append nums[1] to arr1"
    # This is nums[0] in the 0-indexed Python list.
    arr1.append(nums[0])

    # Second operation: "append nums[2] to arr2"
    # This is nums[1] in the 0-indexed Python list.
    arr2.append(nums[1])

    # Subsequent operations:
    # "in the i^th operation: ... append nums[i] ..."
    # This means for operations i = 3, 4, ..., n, we consider the element nums[i] (1-indexed).
    # In Python's 0-indexed list, this corresponds to iterating from index 2 up to n-1
    # and processing nums[python_list_idx].
    n = len(nums)
    for python_list_idx in range(2, n):
        current_element = nums[python_list_idx]
        
        # At this stage (python_list_idx >= 2), arr1 and arr2 are guaranteed to have 
        # at least one element from the first two operations.
        # The constraint n >= 3 ensures this loop body is entered only if there are elements
        # beyond the first two.
        if arr1[-1] > arr2[-1]:
            arr1.append(current_element)
        else:
            # This covers arr1[-1] < arr2[-1].
            # arr1[-1] == arr2[-1] is not possible because all elements in nums are distinct,
            # and each element is added to either arr1 or arr2, but not both.
            arr2.append(current_element)
            
    # Concatenate arr1 and arr2 to form the result array.
    return arr1 + arr2