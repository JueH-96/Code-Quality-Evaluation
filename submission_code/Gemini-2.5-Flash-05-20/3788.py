from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        """
        Calculates the maximum sum of a subarray where all elements are unique.
        
        The problem implies that we can effectively choose any unique elements
        from the original array `nums` to form a subsequence, and then find
        the maximum sum of a "subarray" from this subsequence that has unique elements.
        Since we want to maximize the sum and all chosen elements must be unique,
        this reduces to selecting all positive unique elements from the original `nums`.
        If there are no positive unique elements, we must select the largest (closest to zero)
        among the unique elements (which will be 0 or a negative number) because
        the resulting array cannot be empty.
        
        Args:
            nums: A list of integers.
            
        Returns:
            The maximum sum of a subarray with unique elements.
        """
        
        # Step 1: Get all unique numbers present in the input array.
        # Using a set naturally handles uniqueness and provides O(N) average time
        # for creation from a list.
        unique_nums = set(nums)
        
        # Step 2: Calculate the sum of all unique positive numbers.
        # This is the primary candidate for the maximum sum, as adding any positive
        # unique number increases the sum, and adding any non-positive unique number
        # either keeps it same (for 0) or decreases it (for negative).
        total_sum_of_positives = 0
        for num in unique_nums:
            if num > 0:
                total_sum_of_positives += num
        
        # Step 3: Determine the final maximum sum based on the sum of positive unique numbers.
        if total_sum_of_positives > 0:
            # If we found at least one positive unique number, then the maximum sum
            # is achieved by selecting all such positive unique numbers.
            # For example, if nums = [1, 5, -2, 1], unique_nums = {1, 5, -2}.
            # Positive unique numbers are 1 and 5. Their sum is 6.
            # We can form a subsequence [1, 5] (by deleting -2 and one of the 1s).
            # This subsequence itself is a "subarray" with unique elements and sum 6.
            return total_sum_of_positives
        else:
            # If total_sum_of_positives is 0, it means there are no positive unique numbers.
            # All unique numbers are either zero or negative (e.g., nums = [0, -1, -5] or [-1, -2, -3]).
            # The problem states that after deletions, the array cannot be empty,
            # implying we must select at least one element.
            # To maximize the sum when all unique elements are non-positive,
            # we should pick the single largest value among the unique elements.
            # This will be 0 if 0 is present and largest, or the largest negative number otherwise.
            # Since nums.length >= 1 (per constraints), unique_nums is guaranteed to not be empty,
            # so max(unique_nums) is safe.
            return max(unique_nums)