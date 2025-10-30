from typing import List
import bisect
import math

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        # ------------------------------------------------------------------
        # helper : build powers of four up to a little more than 1e9
        # ------------------------------------------------------------------
        POW4 = [1]
        LIMIT = 10 ** 9                       # maximal r
        while POW4[-1] <= LIMIT:
            POW4.append(POW4[-1] * 4)         # one extra value that is > LIMIT
        LAST_IDX = len(POW4) - 2              # index of the largest power <= LIMIT
        
        # ------------------------------------------------------------------
        # helper : prefix sum F(n) = sum_{x=1..n}  f(x),
        #          where f(x) = number of times x has to be divided by 4
        #                      until it becomes 0
        #          (f(x) = floor(log_4(x)) + 1  for x>0, else 0)
        # ------------------------------------------------------------------
        def prefix_sum(n: int) -> int:
            if n <= 0:
                return 0
            res = 0
            for k in range(LAST_IDX + 1):         # iterate over intervals [4^k , 4^{k+1}-1]
                start = POW4[k]
                if start > n:
                    break
                end = min(n, POW4[k + 1] - 1)
                cnt = end - start + 1             # how many numbers in this segment
                res += (k + 1) * cnt              # every number in the segment has f(x)=k+1
            return res
        
        # ------------------------------------------------------------------
        # main part : evaluate every query
        # ------------------------------------------------------------------
        total_ops = 0
        for l, r in queries:
            # sum of f(x) over [l, r]
            S = prefix_sum(r) - prefix_sum(l - 1)
            
            # maximum f(x) over the interval is obtained at x = r
            # find floor(log_4(r)) quickly via the powers array
            k = bisect.bisect_right(POW4, r) - 1     # largest k with 4^k <= r
            max_f = k + 1                            # f(r)
            
            # minimal operations for this query
            ops = max(max_f, (S + 1) // 2)           # ceil(S/2) = (S+1)//2
            total_ops += ops
        
        return total_ops