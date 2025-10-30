from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        """
        Brute–force is entirely feasible because the array length is ≤ 13.
        There are at most 2¹³-1 = 8 191 non-empty subsets; computing the
        product for each of them (≤ 13 multiplications) is trivial.
        """
        n = len(nums)
        best = None                                   # best product seen so far
        for mask in range(1, 1 << n):                # iterate over all non-empty subsets
            prod = 1
            for i in range(n):                       # build the product of this subset
                if mask & (1 << i):
                    prod *= nums[i]
            if best is None or prod > best:          # keep the maximum
                best = prod
        return best