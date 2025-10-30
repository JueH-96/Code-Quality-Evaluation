from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # If there is only one or two elements we can always (trivially) obtain
        # n single-element arrays – either no split is needed or one split that
        # produces two size-1 sub-arrays (length-1 sub-arrays are always allowed).
        if n <= 2:
            return True

        # For a longer array the task is possible **iff** there exists at least one
        # adjacent pair whose sum is ≥ m.
        #
        # Reasoning (sketch):
        #   • Let i, i+1 be such a pair (nums[i]+nums[i+1] ≥ m).
        #   • Repeatedly split off single elements from the left side of the array
        #     until index i becomes the first element of the remaining segment.
        #     Each split is valid because the right segment still contains the
        #     (i, i+1) pair, hence its sum is ≥ m.
        #   • Do the same on the right side: keep removing one element at a time
        #     from the end until only the pair ({i},{i+1}) is left.  Every removal
        #     is valid because the left segment always contains that pair whose
        #     sum is ≥ m.
        #   • Finally split the length-2 segment into two singletons.  Both are
        #     length-1, so the rule is satisfied.
        #
        #   Hence the existence of such an adjacent pair is sufficient, and it is
        #   clearly necessary (a length-2 segment with sum ≥ m must appear in the
        #   splitting process).  Therefore the criterion below is both necessary
        #   and sufficient.
        for i in range(n - 1):
            if nums[i] + nums[i + 1] >= m:
                return True
        return False