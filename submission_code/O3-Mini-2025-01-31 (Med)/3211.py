from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        # Explanation:
        # We wish to “simulate” a sequence of merge‐operations on contiguous segments.
        # In the end, the array becomes a segmentation of the original array; each segment’s value
        # equals the sum of its numbers. We require the final sequence (of segment sums) to be
        # non‐decreasing. (That is, s1 <= s2 <= ... <= sk.)
        #
        # Equivalently, we want to partition the array into k contiguous groups so that 
        # s1, s2, …, sk – where s_i is the sum of group i – satisfy s1 <= s2 <= … <= sk.
        # Our goal is to maximize k.
        #
        # One might notice that if no operations are applied then the final array is exactly
        # nums. So if nums is already non‐decreasing then k = len(nums) is possible.
        #
        # In general, when nums is not sorted, we must “merge” some adjacent elements together.
        # Merging reduces the number of segments; therefore the answer is less than len(nums).
        #
        # Our idea is to use binary search on k (from 1 to len(nums)) and check feasibility.
        # Given a candidate k, we ask: can we partition nums into k contiguous segments (covering
        # all of nums, though if we use fewer than all elements in the k-th segment, we can always “absorb”
        # the remaining ones into the last segment) such that if s1, s2, …, sk are the sums, then
        # s1 <= s2 <= … <= sk?
        #
        # Because all numbers are positive, when forming a segment starting at index i, if we
        # want the segment sum to be at least L (which is the previous segment’s sum), then:
        #   – Taking too few elements might yield too small a sum and fail the L requirement.
        #   – Taking extra elements only increases the segment sum.
        # To maximize the chance of being able to “save” more segments later, we want the segment
        # sum to be as small as possible while still being >= L.
        #
        # Greedy feasibility algorithm check(k):
        #   Let i = 0 be the starting index for segments, and last = 0 (for the first segment, there is no lower bound,
        #   but since nums are positive, taking a single element will always yield a sum >= 0).
        #   For seg = 0 to k-1:
        #     - Determine the maximum index we are allowed to use for the current segment.
        #       (We must leave at least (k - seg - 1) elements, one per future segment.)
        #       For seg < k-1, allowed_max = len(nums) - (k - seg - 1)
        #       For seg == k-1, we can use the rest, so allowed_max = len(nums)
        #     - Starting from index i, accumulate a running sum.
        #       Let j iterate from i to allowed_max - 1.
        #       Find the earliest j where running sum >= last.
        #       If no such j exists, then partitioning into k segments is impossible.
        #     - Set last equal to the found segment sum, and update i = j+1.
        #   If we finish (even if i < len(nums), the leftover can be merged with the last segment)
        #   then the candidate k is feasible.
        #
        # Then we binary search for the maximum k from 1 to len(nums) for which check(k) returns True.
        
        n = len(nums)
        
        # Quick check: if the original array is already non-decreasing,
        # we can keep all elements.
        is_non_decr = True
        for i in range(1, n):
            if nums[i-1] > nums[i]:
                is_non_decr = False
                break
        if is_non_decr:
            return n
        
        # Feasibility function:
        def can_partition(k: int) -> bool:
            i = 0
            last = 0  # lower bound for the segment sum; for the first segment, 0 works
            # Try to form k segments.
            for seg in range(k):
                # Determine the maximum index we can use for this segment.
                # Must leave at least (k - seg - 1) elements.
                if seg < k - 1:
                    allowed_max = n - (k - seg - 1)
                else:
                    allowed_max = n
                cur_sum = 0
                found = False
                # j goes from i to allowed_max - 1
                for j in range(i, allowed_max):
                    cur_sum += nums[j]
                    # If current running sum satisfies the non-decrease requirement over the previous segment:
                    if cur_sum >= last:
                        # We choose to cut here.
                        found = True
                        i = j + 1
                        last = cur_sum
                        break
                if not found:
                    return False
            # We do not require to use all elements exactly,
            # because any leftover can be merged into the last segment.
            return True
        
        # Binary search for maximum k in [1, n].
        lo, hi = 1, n
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_partition(mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        return best