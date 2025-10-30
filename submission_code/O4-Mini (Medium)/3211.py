from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        # Check if it's possible to split into k non-decreasing-sum segments
        def feasible(k: int) -> bool:
            # k segments total, each contiguous, sums non-decreasing
            # We greedily make the first k-1 segments as short as possible
            # so that each segment sum >= previous segment sum.
            if k == 1:
                # One segment: the whole array
                return True
            pos = 0
            last_sum = 0
            consumed = 0  # sum of elements consumed in first k-1 segments
            # build first k-1 segments
            for _ in range(k - 1):
                curr_sum = 0
                # accumulate until we reach at least last_sum
                while pos < n and curr_sum < last_sum:
                    curr_sum += nums[pos]
                    pos += 1
                if curr_sum < last_sum:
                    # couldn't reach required sum
                    return False
                last_sum = curr_sum
                consumed += curr_sum
            # Last segment is the rest of the array
            if pos >= n:
                # no elements left for the last segment
                return False
            last_seg_sum = total_sum - consumed
            return last_seg_sum >= last_sum
        
        # Binary search for the maximum k in [1..n]
        lo, hi = 1, n
        ans = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans