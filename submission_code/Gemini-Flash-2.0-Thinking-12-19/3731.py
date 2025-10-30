from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        """
        Calculates the total sum of elements from specific subarrays defined for each index.

        For each index i (0 <= i < n), a subarray nums[start ... i] is defined
        where start = max(0, i - nums[i]). This function iterates through
        each index i, calculates the sum of the corresponding subarray,
        and returns the total sum of all these subarray sums.

        Args:
            nums: The input list of integers of size n.

        Returns:
            The total sum of all elements from the defined subarrays.
        """
        n = len(nums)
        total_sum = 0

        # Iterate through each index i from 0 to n-1
        for i in range(n):
            # Define the start index for the subarray nums[start ... i]
            # start = max(0, i - nums[i])
            # nums[i] is guaranteed to be >= 1, so i - nums[i] can be negative.
            # max(0, ...) ensures the start index is never less than 0.
            start = max(0, i - nums[i])

            # The subarray for index i consists of elements from nums[start] up to nums[i], inclusive.
            # In Python slicing, this corresponds to nums[start : i + 1].
            
            # Calculate the sum of this specific subarray.
            # Using slicing and the built-in sum() function is a concise way.
            # This step contributes to the O(N^2) complexity in the worst case
            # as the slice can be up to length N.
            current_subarray_sum = sum(nums[start : i + 1])

            # Add the sum of the current subarray to the running total.
            total_sum += current_subarray_sum

        return total_sum