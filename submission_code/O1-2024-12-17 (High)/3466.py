from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        """
        We want to find all subarrays such that the bitwise AND of
        all elements in the subarray equals k.

        Observations:
        1) If a subarray's AND is to be exactly k, then for every bit that is
           1 in k, no element in the subarray can have that bit = 0 (otherwise
           the AND would lose that bit).
           => Any element that has a 0 in a bit where k has a 1 immediately
              invalidates that subarray. We call such elements "invalid."
        2) For each bit that is 0 in k, at least one element in the subarray
           must have that bit = 0 (otherwise that bit would stay 1 in the AND).

        Approach:
        1) Split nums into segments separated by "invalid" elements. An element
           x is invalid if (x & k) != k, meaning x has at least one bit = 0
           where k has that same bit = 1. No valid subarray can cross such an
           element.
        2) Within each valid segment, we only have elements that keep all bits
           in k intact (no "forbidden" zeros). To ensure the AND is exactly k,
           we must ensure that for each bit that is 0 in k, our subarray picks
           up at least one element with a 0 in that bit.
        3) To count subarrays in a segment [s..e] that have those zero-bits
           covered, do a standard coverage technique:
           - Let bits0 be the list of bit positions i where k has bit i = 0.
           - Maintain an array last0 of the same length as bits0, initially [-1].
             last0[j] will store the last index r where nums[r] has bit bits0[j] = 0.
           - Iterate r from s to e. For each nums[r], update last0 for any bit
             that is 0 in nums[r]. Let minLast0 = min(last0). If minLast0 >= s
             (i.e., all bits in bits0 have been found at least once in [s..r]),
             then every subarray ending at r that starts in [s..minLast0] is valid.
             The number of new valid subarrays ending at r is (minLast0 - s + 1).

        The total of all those counts across all valid segments is our answer.
        """

        # Collect which bits are 0 in k (we need to cover these bits with zeros)
        bits0 = [i for i in range(32) if not (k & (1 << i))]

        def count_in_segment(start: int, end: int) -> int:
            """
            Counts how many subarrays within nums[start..end] have AND == k,
            assuming none of these elements are invalid for the '1-bits' of k.
            """
            if start > end:
                return 0

            # last0[j] = last index where we found a 0 for bit bits0[j]
            last0 = [-1] * len(bits0)
            total_sub = 0

            left_boundary = start
            for r in range(start, end + 1):
                val = nums[r]
                # Update last0 for zero-bits
                for j, b in enumerate(bits0):
                    if (val & (1 << b)) == 0:
                        last0[j] = r

                # Check if we have coverage for all bits0 (no -1 in last0)
                if -1 in last0:
                    # Not all bits covered yet
                    continue

                # We have coverage: subarray ends at r and must start at or before
                # min_last to include all zero-bits
                min_last = min(last0)
                total_sub += (min_last - left_boundary + 1)

            return total_sub

        total = 0
        n = len(nums)

        # We'll split nums by invalid elements where (nums[i] & k) != k
        seg_start = 0
        for i in range(n):
            if (nums[i] & k) != k:
                # process previous segment
                total += count_in_segment(seg_start, i - 1)
                seg_start = i + 1

        # last segment
        if seg_start < n:
            total += count_in_segment(seg_start, n - 1)

        return total