from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # full bits up to bit 29
        ALL_BITS = (1 << 30) - 1
        
        # check if we can achieve final OR <= mask with at most k merges
        def can(mask: int) -> bool:
            forbidden = ALL_BITS & ~mask
            ops = 0
            i = 0
            while i < n:
                # try to form the shortest segment starting at i whose AND clears forbidden bits
                cur_and = ALL_BITS
                found = False
                for j in range(i, n):
                    cur_and &= nums[j]
                    # if this segment AND has no forbidden bits, we can end here
                    if (cur_and & forbidden) == 0:
                        ops += (j - i)  # merges needed
                        i = j + 1
                        found = True
                        break
                    # early exit if ops exceed k
                    if ops > k:
                        return False
                if not found:
                    # no valid segment starting at i
                    return False
                if ops > k:
                    return False
            return ops <= k

        # build answer bit by bit, trying to keep bits zero if possible
        ans = 0
        # from highest bit down to 0
        for b in reversed(range(30)):
            # allow all lower bits provisionally
            provisional = ans | ((1 << b) - 1)
            # test if we can avoid setting bit b
            if not can(provisional):
                # we must set bit b
                ans |= (1 << b)
        return ans