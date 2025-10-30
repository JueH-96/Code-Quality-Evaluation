from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        """
        Calculates the maximum number of subarrays a given array can be split into
        such that the sum of their bitwise AND scores is minimized.

        Args:
            nums: A list of non-negative integers.

        Returns:
            The maximum number of subarrays in a split achieving the minimum sum of scores.
        """
        n = len(nums)
        if n == 0:
            return 0 # Constraint 1 <= n, but good practice

        # Calculate the bitwise AND of the entire array.
        # This value represents the minimum possible score for any single subarray
        # covering a contiguous part of the array.
        total_and = nums[0]
        for i in range(1, n):
            total_and &= nums[i]

        # If the bitwise AND of the entire array is greater than 0,
        # the minimum possible sum of scores is equal to this total_and.
        # This minimum sum can only be achieved by taking the entire array
        # as a single subarray. If we split into multiple subarrays, each
        # subarray's score would be >= total_and (since all numbers contain
        # the bits set in total_and), leading to a sum >= number_of_subarrays * total_and.
        # For more than one subarray, this sum would be strictly greater than total_and.
        # Thus, if total_and > 0, the maximum number of subarrays achieving the
        # minimum sum is 1.
        if total_and > 0:
            return 1

        # If the bitwise AND of the entire array is 0,
        # the minimum possible sum of scores is 0.
        # This sum is achieved if and only if every subarray in the split
        # has a score of 0.
        # We want to find the maximum number of subarrays such that each has a score of 0.
        # We use a greedy approach: iterate through the array, maintaining the bitwise AND
        # of the current potential subarray starting from the last cut point.
        # When this current segment's AND becomes 0, we finalize this segment as a subarray,
        # increment the count of subarrays, and start considering a new potential subarray
        # from the next element. This strategy maximizes the number of score-0 subarrays
        # that can partition the array.
        
        count = 0
        current_and_segment = None # Use None to indicate we are starting a new segment

        for num in nums:
            if current_and_segment is None:
                # Start a new segment with the current number
                current_and_segment = num
            else:
                # Extend the current segment by ANDing with the current number
                current_and_segment &= num

            if current_and_segment == 0:
                # We found a segment ending at the current position with score 0.
                # We make this a complete subarray and are ready to start a new one
                # from the next element. This contributes to the minimum sum (0).
                count += 1
                current_and_segment = None # Reset to start a new segment

        # If total_and was 0, the minimum sum is 0. The greedy process finds
        # the maximum number of segments that can be formed, each having a score of 0.
        # This set of segments forms a partition of the array.
        # Since total_and is 0, the entire array has score 0, guaranteeing that
        # at least one score-0 segment can be formed (the whole array itself).
        # The greedy algorithm finds the maximum number of such segments by cutting
        # as early as possible whenever a score-0 segment is found.
        # The count will be at least 1 if nums is not empty (due to total_and == 0).
        # The case where count is 0 when total_and is 0 is impossible for non-empty nums,
        # as the cumulative AND starting from index 0 must become 0 by the end of the array.
        
        return count