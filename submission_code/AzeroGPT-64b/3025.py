from collections import Counter
from functools import lru_cache

class Solution:
    def minOperations(self, nums, target):
        bitcount = Counter()
        for x in nums:
            bitcount[31 - x.bit_length()] += 1

        target_cnts = Counter()
        while target:
            target_cnts[target.bit_length() - 1] += 1
            target &= target - 1

        @lru_cache(None)
        def dp(i, count):
            if count >= 0 and i >= 0:
                return 0

            if i < 0:
                return float('inf')

            # case 1: skip i'th position and move on to next.
            res1 = dp(i - 1, count)

            # case 2: consume a number whose bit length is i to satisfy target,
            #         the leftover is divided into the next two level (i - 1, i - 2)
            res2 = float('inf')
            if target_cnts[i] > count:
                needed = target_cnts[i] - count
                if bitcount[i] >= needed:
                    res2 = needed + dp(i - 1, count - needed) + \
                           dp(i - 2, count + bitcount[i] - needed)

            return min(res1, res2)

        res = dp(30, 0)
        return res if res < float('inf') else -1