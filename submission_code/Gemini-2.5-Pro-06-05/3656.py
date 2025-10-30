from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of operations to make array elements distinct.

        An operation consists of removing 3 elements from the beginning of the array.
        This problem can be solved by finding the longest suffix of the array that
        contains distinct elements. The elements before this suffix must be removed.

        Args:
            nums: The input list of integers.

        Returns:
            The minimum number of operations required.
        """
        
        # We can find the start of the longest distinct suffix by iterating
        # from the right of the array, using a set to track seen elements.
        n = len(nums)
        seen_in_suffix = set()
        
        # Iterate from the last element to the first.
        for i in range(n - 1, -1, -1):
            current_element = nums[i]
            
            # If we've already seen this element in the suffix, it's a duplicate.
            if current_element in seen_in_suffix:
                # The duplicate is at index `i`. This means the longest
                # distinct suffix must start at the next index, `i + 1`.
                # The prefix to be removed has a length of `i + 1`.
                prefix_length_to_remove = i + 1
                
                # The number of operations is ceil(prefix_length_to_remove / 3).
                # Ceiling division can be calculated using integer arithmetic:
                # (numerator + denominator - 1) // denominator
                return (prefix_length_to_remove + 2) // 3
            
            # If the element is new to the suffix, add it to our set.
            seen_in_suffix.add(current_element)
            
        # If the loop completes, the entire array has distinct elements.
        # No operations are needed.
        return 0