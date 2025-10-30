from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Sort the array numerically
        sorted_nums = sorted(nums)
        n = len(nums)

        # Compute popcounts for original and sorted arrays
        # Using Python 3.8+'s int.bit_count(); fallback to bin().count if needed
        try:
            pop = int.bit_count
        except AttributeError:
            pop = lambda x: bin(x).count('1')

        orig_bits = [pop(x) for x in nums]
        sorted_bits = [pop(x) for x in sorted_nums]

        # If the bit‐count pattern differs, we can never match the sorted array
        if orig_bits != sorted_bits:
            return False

        # Within each contiguous run of equal bit‐counts, we can only
        # permute among those positions.  Check that each run in the
        # original contains exactly the same multiset of values as in
        # the target sorted array.
        i = 0
        while i < n:
            j = i
            # extend the run while the bit‐count stays the same
            while j + 1 < n and orig_bits[j+1] == orig_bits[i]:
                j += 1

            # slice out the two segments
            seg_orig = nums[i:j+1]
            seg_target = sorted_nums[i:j+1]

            # if their multisets differ, sorting is impossible
            if sorted(seg_orig) != seg_target:
                return False

            i = j + 1

        # All checks passed: we can achieve the fully sorted array
        return True