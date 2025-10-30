from collections import deque
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        """
        Counts contiguous sub-arrays whose elements all differ from each other
        by at most 2 (equivalently, sub-arrays whose max – min ≤ 2).
        Sliding-window + two monotone deques keep current window’s min and max.
        """
        min_dq = deque()        # increasing: front is current window minimum
        max_dq = deque()        # decreasing: front is current window maximum
        left = 0
        res = 0

        for right, val in enumerate(nums):
            # update deques with new value
            while min_dq and val < min_dq[-1]:
                min_dq.pop()
            min_dq.append(val)

            while max_dq and val > max_dq[-1]:
                max_dq.pop()
            max_dq.append(val)

            # shrink window until condition satisfied
            while max_dq[0] - min_dq[0] > 2:
                if nums[left] == min_dq[0]:
                    min_dq.popleft()
                if nums[left] == max_dq[0]:
                    max_dq.popleft()
                left += 1

            # all subarrays ending at 'right' and starting >= left are valid
            res += right - left + 1

        return res