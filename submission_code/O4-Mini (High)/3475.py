from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # flips[i] = 1 if we applied a flip operation starting at i
        flips = [0] * n
        curr_flip = 0  # number of active flips affecting current index
        ops = 0

        for i in range(n):
            # remove the effect of the flip that started 3 positions ago
            if i >= 3:
                curr_flip -= flips[i - 3]
            
            # After curr_flip flips, the current bit is (nums[i] + curr_flip) % 2.
            # We want it to be 1. If it is 0, we must flip at i (covering i, i+1, i+2).
            if (nums[i] + curr_flip) % 2 == 0:
                # If we can't flip a full window of length 3, it's impossible
                if i + 2 >= n:
                    return -1
                # Mark a flip at i
                flips[i] = 1
                curr_flip += 1
                ops += 1

        return ops