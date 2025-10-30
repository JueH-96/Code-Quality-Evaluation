from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10 ** 9 + 7
        n = len(nums)
        
        # For each number, record its last occurrence.
        last = {}
        for i, num in enumerate(nums):
            last[num] = i

        # The key observation:
        # A partition (into contiguous subarrays) is "good" if no element appears
        # in more than one subarray. That is equivalent to not “cutting” an interval 
        # [first_occurrence, last_occurrence] of any number.
        #
        # Notice that a necessary and sufficient condition for inserting a cut 
        # immediately after index i (i.e. between i and i+1) is that no number
        # that appears in nums[0:i+1] will appear later. Equivalently, if we maintain 
        # a running maximum 'current_max' which is the maximum over last[num] for each num
        # seen so far, then a cut is allowed after index i exactly when i equals current_max.
        #
        # Now, if you scan the array from left to right, you can “force” a cut at every
        # index i where i == current_max. These indices yield the minimal segmentation 
        # such that no element’s occurrences cross a block boundary (this is sometimes 
        # known as the "partition labels" method).
        #
        # Finally, note that if the minimal segmentation divides the array into k blocks,
        # then any good partition must be obtained by merging some adjacent blocks.
        # There are exactly k - 1 merging choices (between consecutive blocks),
        # and each such cut can either be kept or removed.
        #
        # Therefore, the total number of good partitions is 2^(k - 1) (with k at least 1).
        
        segments = 0
        current_max = 0
        for i, num in enumerate(nums):
            current_max = max(current_max, last[num])
            if i == current_max:
                segments += 1  # We found a boundary where a cut is allowed.
        
        # There is always at least one segment, so the answer is 2^(segments - 1)
        return pow(2, segments - 1, MOD)