from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        """
        For every value v (v != k) treat each array element as
            +1  if nums[i] == v          (this element can be turned into k)
            -1  if nums[i] == k          (this element would be lost if it lies inside the segment)
             0  otherwise
        The best segment for this v is the maximum sub-array sum of that
        transformed array.  It gives the largest possible improvement
        (#v inside segment  –  #k inside segment).

        The answer is   original number of k's  +  best positive improvement
        obtained for some v (or 0 if no positive improvement exists).

        Complexity :  50 · n   ( because values are in [1 … 50] ),  n ≤ 1e5.
        """
        # current amount of k's in the original array
        base = sum(1 for x in nums if x == k)
        best_delta = 0                     # best additional k's we can obtain

        for v in range(1, 51):             # values are between 1 and 50
            if v == k:
                continue

            cur = 0                        # Kadane's current sum
            best = 0                       # best sum for this v

            for val in nums:
                if val == v:
                    cur += 1               # this position can become k
                elif val == k:
                    cur -= 1               # this k would be destroyed
                # else: contribution is 0

                if cur < 0:                # restart the segment
                    cur = 0
                elif cur > best:           # track best improvement
                    best = cur

            if best > best_delta:
                best_delta = best

        return base + best_delta