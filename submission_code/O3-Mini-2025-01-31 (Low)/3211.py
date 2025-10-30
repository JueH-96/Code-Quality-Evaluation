from typing import List
import bisect

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        # The key observation is that any sequence of operations (each replacing a contiguous subarray
        # with its sum) amounts to partitioning the original array into contiguous segments.
        # The value of each segment is the sum of its elements.
        # Our goal is to partition the array into as many segments as possible such that
        # if the segments are s1, s2, ..., sk their sums satisfy s1 <= s2 <= ... <= sk.
        #
        # We precompute the prefix sums of the array. Notice that since all numbers are positive,
        # the prefix sum array is strictly increasing. For any segment covering indices [i, j]
        # the segment’s sum is prefix[j+1] – prefix[i].
        #
        # Thus, the problem is reformulated as:
        #   Partition the array (from index 0 to n-1) into as many segments as possible
        #   such that for consecutive segments (say segment1 from [0, i-1] and segment2 from [i, j]),
        #   we have:
        #     (prefix[i] - prefix[0]) <= (prefix[j+1] - prefix[i])
        #   If we let 'prev' be the sum of the previous segment we have chosen,
        #   then for the current segment starting at index i, we want:
        #       (prefix[k] - prefix[i]) >= prev
        #   and ideally, we want to choose the smallest possible segment meeting this requirement
        #   so that future segments can be as “small” as possible and allow more segments.
        #
        # The plan:
        #   - Precompute prefix sum array P, where P[0] = 0 and P[i+1] = P[i] + nums[i]
        #   - Let curr_index = 0, and let prev = 0 (the previous segment sum).
        #   - At each step, we have a segment starting at index i.
        #     We need to find the smallest j (with j in [i+1, n]) such that:
        #         P[j] - P[i] >= prev   (or, equivalently, P[j] >= prev + P[i])
        #     Because P is increasing, we can binary search to find such a j.
        #   - Update prev to be the current segment sum, and set next segment's start index to j.
        #   - Count each segment we successfully form.
        #   - The maximum possible length of the non-decreasing array after operations is the number of segments.
        #
        # Time complexity: O(n log n) which is acceptable for n up to 10^5.

        n = len(nums)
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
            
        prev = 0  # previous segment's sum
        count = 0
        i = 0  # starting index for the next segment
        while i < n:
            # We want minimal k in [i+1, n] satisfying:
            #    prefix[k] >= prev + prefix[i]
            target = prev + prefix[i]
            # Use binary search on prefix, note that prefix is sorted.
            # bisect_left returns the index of the leftmost value >= target.
            k = bisect.bisect_left(prefix, target, i+1, n+1)
            if k == n+1 or k > n or k < 0 or k == 0:
                break
            # The current segment is from i to k-1
            seg_sum = prefix[k] - prefix[i]
            # update for next iteration
            prev = seg_sum
            i = k
            count += 1
        return count