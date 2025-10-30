from typing import List
from collections import Counter

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        """
        Finds the largest integer that appears in exactly one subarray of size k.

        An integer x is almost missing from nums if x appears in exactly one
        subarray of size k within nums. A subarray is a contiguous sequence
        of elements.

        Args:
            nums: The input integer array.
            k: The size of the subarrays to consider.

        Returns:
            The largest almost missing integer, or -1 if no such integer exists.
        """
        n = len(nums)
        
        # A Counter to store how many subarrays of size k each number appears in.
        # The keys are the numbers, and the values are the counts of k-sized
        # subarrays containing that number at least once.
        # Counter handles initialization to 0 automatically when a new key is accessed.
        subarray_appearance_counts = Counter()

        # Iterate through all possible starting indices for a subarray of size k.
        # A subarray of size k starting at index i goes from nums[i] to nums[i+k-1].
        # The first possible start index is 0.
        # The last possible start index is when the end of the subarray
        # (index i + k - 1) is the last index of the array (n - 1).
        # So, i + k - 1 = n - 1 => i = n - k.
        # The range of start indices is from 0 up to n - k, inclusive.
        for i in range(n - k + 1):
            # Extract the current subarray of size k using slicing.
            current_subarray = nums[i : i + k]
            
            # Get the set of unique elements within the current subarray.
            # We only care if a number is present in a subarray, not how many times
            # it's repeated within that specific subarray. Using a set gives us the
            # distinct values present.
            unique_elements_in_current_subarray = set(current_subarray)
            
            # For each unique element found in the current subarray, increment its count
            # in our main counter. This counter tracks how many *distinct* k-sized
            # subarrays contain this element.
            for x in unique_elements_in_current_subarray:
                subarray_appearance_counts[x] += 1
        
        # Now, we need to find the largest integer 'x' such that its count
        # in subarray_appearance_counts is exactly 1.
        # This means 'x' appeared in exactly one subarray of size k.
        
        # The problem constraints state that 0 <= nums[i] <= 50.
        # This implies that any number present in the input array, and thus
        # potentially in its subarrays, must be within this range [0, 50].
        # To find the *largest* such integer efficiently, we can iterate
        # through the possible integer values from the maximum (50) down to
        # the minimum (0). The first number we encounter that has a count of 1
        # in our `subarray_appearance_counts` is guaranteed to be the largest
        # such integer.
        
        # Iterate from 50 down to 0 (inclusive).
        for x in range(50, -1, -1):
            # Check if the number 'x' exists as a key in our counter
            # and if its corresponding count is exactly 1.
            # Using `.get(x, 0)` is a safe way to check the count. If 'x' is not
            # in the counter at all, `.get(x, 0)` returns 0, meaning 'x' appeared
            # in 0 subarrays, which is not equal to 1. If 'x' is in the counter,
            # it returns its actual count.
            if subarray_appearance_counts.get(x, 0) == 1:
                # We found the largest integer 'x' that satisfies the condition
                # (appears in exactly one subarray of size k).
                return x

        # If the loop finishes without finding any integer with a count of 1,
        # it means no such integer exists in the specified range (0-50) that meets
        # the condition. In this case, according to the problem statement, return -1.
        return -1