from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        """
        Scan the array once while keeping track of the length of the current
        alternating sub-array that ends at the current position.

        A valid alternating sub-array must
        1. have length ≥ 2, and
        2. start with a +1 difference, afterwards the differences must
           alternate between -1 and +1.

        cur_len  – length of the current candidate sub-array
        prev_dif – the last difference (+1 / -1) inside this candidate

        Whenever the rule breaks we start a new candidate if the fresh
        difference equals +1, otherwise we reset.
        """
        n = len(nums)
        # longest length found so far (-1 means “none found yet”)
        best = -1        
        
        cur_len = 1          # current sub-array length (at least the current element)
        prev_dif = None      # last stored difference (+1 / -1)

        for i in range(1, n):
            dif = nums[i] - nums[i - 1]

            if dif == 1:                       # a +1 difference
                if prev_dif == -1:             # alternates correctly
                    cur_len += 1
                else:                          # either first +1 or two +1’s in a row
                    cur_len = 2                # (re-)start from the previous element
                prev_dif = 1

            elif dif == -1 and prev_dif == 1:  # correct “-1” that alternates
                cur_len += 1
                prev_dif = -1

            else:                              # pattern broken
                cur_len = 1
                prev_dif = None

            if cur_len > 1:
                best = max(best, cur_len)

        return best