from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        """
        Let diff[i] = target[i] – nums[i].
        While scanning diff from left to right we keep the previous value `prev`.

        • If the current value is larger than `prev`, the array has to be raised by
          (cur - prev) starting exactly at this position, therefore we need that
          many new “+1-on-a-segment” operations.

        • If the current value is smaller than `prev`, the array has to be lowered by
          (prev - cur) starting here, therefore we need that many new “-1-on-a-segment”
          operations.

        Counting those starts separately gives:
              inc = total number of +1 operations that must start
              dec = total number of -1 operations that must start

        Any operation can finish later for free (it was already counted when it
        started), so the total amount of operations we really have to perform is
        the larger of the two counters.
        """
        inc = dec = 0         # how many +1 / -1 operations must be started
        prev = 0              # diff value at the previous index

        for a, b in zip(nums, target):
            cur = b - a       # current diff
            if cur > prev:    # need (cur - prev) new +1 operations
                inc += cur - prev
            else:             # need (prev - cur) new -1 operations
                dec += prev - cur
            prev = cur

        return max(inc, dec)