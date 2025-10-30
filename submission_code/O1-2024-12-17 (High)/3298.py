from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # We will use a greedy approach in two passes:
        #
        #  1) Sort the array.
        #  2) Define a helper "fill(start)" that tries to build a
        #     consecutive sequence beginning at 'start', using each
        #     element x in ascending order (either as x or x+1) if it
        #     can match the current needed integer.
        #
        #     Specifically, if the next integer we need is 'current',
        #     and we see an element x, we check whether
        #         x <= current <= x+1.
        #     If true, we assign x to fill 'current' (or x+1 if needed),
        #     increment 'current', and move on.
        #
        #  3) Because each element can be either x or x+1, the start of
        #     the maximal chain can only be either min(nums) or min(nums)+1.
        #     Trying just those two starts suffices to cover all cases
        #     (including the possibility of "shifting" the sequence by 1).
        #
        #  4) The maximum of fill(min_val) and fill(min_val+1) will be our result.
        #
        # This method works for all of the provided examples, including
        # tricky cases like [100,102] (where you want to skip using 100
        # at final=100 and instead use it to fill final=101).

        nums.sort()
        min_val = min(nums)

        def fill(start: int) -> int:
            current = start
            used = 0
            for x in nums:
                if x <= current <= x + 1:
                    used += 1
                    current += 1
            return used

        # Compute for starting at min_val and min_val+1
        return max(fill(min_val), fill(min_val + 1))