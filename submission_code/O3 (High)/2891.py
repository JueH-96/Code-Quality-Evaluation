from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        """
        Each element nums[i] can be moved anywhere inside the interval
        [nums[i] - k, nums[i] + k].  A set of indices can all be turned
        into the same value iff the intersection of all their intervals
        is non-empty.

        For a group of numbers this intersection exists exactly when
            max(nums) - min(nums) ≤ 2·k
        because
            max(nums) - k ≤ min(nums) + k  ⇔  max(nums) - min(nums) ≤ 2k.

        Hence the task reduces to finding the largest subset whose
        maximum and minimum differ by at most 2·k.  After sorting the
        array this is the classical “longest window with bounded
        difference” problem, solvable with a two-pointer / sliding
        window in O(n) time.
        """
        nums.sort()
        ans = 0
        left = 0

        for right, value in enumerate(nums):
            # shrink the window until the condition holds
            while value - nums[left] > 2 * k:
                left += 1
            ans = max(ans, right - left + 1)

        return ans