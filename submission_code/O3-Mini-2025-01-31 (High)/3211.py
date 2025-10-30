from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0

        # The idea is to “partition” the array into segments.
        # Each segment, when replaced by its sum, yields the final array.
        # We want that final array (the sequence of segment sums) to be non-decreasing.
        # To maximize the number of segments (i.e. the final array’s length)
        # we should “cut” each segment as soon as its sum is as small as it can be.
        #
        # Once we fix the first segment, every subsequent segment must have a sum at least
        # as high as the previous segment’s sum. Since all numbers are positive,
        # the best way to achieve this is to take the smallest possible segment that meets this requirement.
        #
        # For example, consider nums = [4, 3, 2, 6]:
        #   - We choose the first segment as just [4] (sum = 4).
        #   - Then we must form a contiguous segment starting at index 1 whose sum is at least 4.
        #     We cannot take [3] because 3 < 4; but [3, 2] has sum = 5 ≥ 4.
        #   - Next, starting after index 2 we have [6] which is ≥ 5, so we can cut.
        # This yields segments: [4], [3,2], [6] and the final array [4,5,6] is non-decreasing.
        #
        # The greedy strategy below implements this idea.
 
        # For the first segment, we must use the first element alone,
        # because enlarging it would only increase its sum and hinder our goal.
        seg_count = 1
        last_segment_sum = nums[0]
        i = 1

        # Try to form as many segments as possible from left to right.
        while i < n:
            current_sum = 0
            # Build the next segment from the current index until:
            # (a) we can get a segment sum at least as large as the previous segment,
            # (b) or we exhaust the array.
            while i < n and current_sum < last_segment_sum:
                current_sum += nums[i]
                i += 1
            # Only if we succeed in reaching at least last_segment_sum, we can form a new segment.
            if current_sum >= last_segment_sum:
                seg_count += 1
                last_segment_sum = current_sum
            # Otherwise, if the remainder (if any) cannot reach the needed sum,
            # we cannot form an extra segment – the leftover must be merged with the previous segment.
        return seg_count