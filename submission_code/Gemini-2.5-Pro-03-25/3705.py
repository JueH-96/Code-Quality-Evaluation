import collections
from typing import List

class Solution:
    """
    Solves the problem of finding the largest "almost missing" integer.
    An integer x is almost missing if it appears in exactly one subarray of size k within nums.
    """
    def largestInteger(self, nums: List[int], k: int) -> int:
        """
        Finds the largest integer that appears in exactly one subarray of size k.

        Args:
            nums: The input list of integers.
            k: The size of the subarrays to consider.

        Returns:
            The largest almost missing integer, or -1 if none exists.
        """
        
        n = len(nums)
        
        # Constraints state 1 <= k <= nums.length, so k > n is not expected,
        # but including the check doesn't hurt. If k > n, there are no subarrays
        # of size k, so the result should arguably be -1 anyway, which the
        # code would naturally produce.
        # if k > n:
        #     return -1 

        # Dictionary to store the count of k-sized subarrays each number appears in.
        # Keys are the distinct numbers encountered in the subarrays, values are their counts.
        # Using collections.defaultdict(int) simplifies adding counts for new numbers.
        subarray_counts = collections.defaultdict(int) 

        # Check if there are any subarrays of size k. The number of such subarrays is n - k + 1.
        # If n - k + 1 <= 0 (i.e., k > n), this loop won't run.
        # Iterate through all possible starting indices for a subarray of size k.
        # The starting indices range from 0 up to and including n - k.
        for i in range(n - k + 1):
            # Extract the current subarray of size k using list slicing.
            # Slicing takes O(k) time.
            current_subarray = nums[i : i + k]
            
            # Find the unique elements within this specific subarray.
            # The problem defines "appears in" based on presence, not frequency within the subarray.
            # Creating a set eliminates duplicates within the current subarray and takes O(k) time on average.
            unique_elements_in_subarray = set(current_subarray)
            
            # For each unique number found in the current subarray, increment its
            # overall count in the `subarray_counts` dictionary. This indicates that
            # this number was found in one more subarray of size k.
            # This loop runs at most k times. Dictionary updates take O(1) time on average.
            for element in unique_elements_in_subarray:
                subarray_counts[element] += 1
        
        # After iterating through all k-sized subarrays and counting occurrences,
        # we need to find the largest number that has a count of exactly 1.
        # Initialize the result variable to -1. This value will be returned if 
        # no number is found with a count of exactly 1.
        max_almost_missing = -1
        
        # Iterate through the items (number, count pairs) stored in the `subarray_counts` dictionary.
        # The number of items is at most the number of unique elements in the original `nums` array.
        for num, count in subarray_counts.items():
            # Check if this number appeared in exactly one subarray of size k.
            if count == 1:
                # If the count is exactly 1, this number is "almost missing".
                # We need the largest such number, so we update `max_almost_missing`
                # if the current number `num` is greater than the maximum found so far.
                max_almost_missing = max(max_almost_missing, num)
                
        # Return the final result: the largest number found with a count of 1,
        # or -1 if no such number was found.
        return max_almost_missing