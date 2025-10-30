from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        """
        For every distinct value count how many times it appears (cnt).
        We have to express cnt as a sum of 2-sized and 3-sized groups while
        minimising the amount of groups (operations).

        Greedy rule (proved optimal):
          • Use as many triples as possible.
          • If exactly one element would be left over (cnt % 3 == 1) we
            replace one triple by two pairs, turning “… + 3 + 1” into
            “… + 2 + 2”.

        Implementation of the above:
          let k, r = divmod(cnt, 3)        # cnt = 3k + r,  r ∈ {0,1,2}
          if r == 0 : ops = k
          if r == 1 : ops = k - 1 + 2 = k + 1   (requires k ≥ 1 → cnt ≥ 4)
          if r == 2 : ops = k + 1
        Any cnt == 1 makes the whole task impossible.
        """
        freq = Counter(nums)
        operations = 0

        for cnt in freq.values():
            if cnt == 1:                     # cannot delete a single element
                return -1

            k, r = divmod(cnt, 3)

            if r == 0:                       # perfect multiples of 3
                operations += k
            elif r == 1:                     # need to convert one triple to two pairs
                if k == 0:                   # would mean cnt == 1 (already ruled out)
                    return -1
                operations += k + 1
            else:                            # r == 2, add one pair
                operations += k + 1

        return operations