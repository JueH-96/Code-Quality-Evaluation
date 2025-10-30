from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2
        # red1[X] = number of pairs for which target X requires at most 1 change (i.e., X <= T_i)
        # red0[X] = number of pairs for which target X requires 0 changes (i.e., X == d_i)
        red1 = [0] * (k + 2)
        red0 = [0] * (k + 1)
        
        for i in range(m):
            a = nums[i]
            b = nums[n - 1 - i]
            if a > b:
                a, b = b, a
            d = b - a
            # T = max(b, k - a)
            T = b if b > (k - a) else (k - a)
            # for X in [0..T], we can fix the pair with at most 1 change
            red1[0] += 1
            red1[T + 1] -= 1
            # for X == d, we need 0 changes
            red0[d] += 1
        
        # build prefix sum of red1
        for X in range(1, k + 1):
            red1[X] += red1[X - 1]
        
        # compute minimal total cost over X = 0..k
        # each of m pairs costs 2 changes initially,
        # we subtract red1[X] (those pairs need at most 1 change => saved 1 per pair)
        # and subtract red0[X] (those pairs need 0 change => saved one more per pair)
        base = 2 * m
        ans = base
        for X in range(k + 1):
            cost = base - red1[X] - red0[X]
            if cost < ans:
                ans = cost
        
        return ans