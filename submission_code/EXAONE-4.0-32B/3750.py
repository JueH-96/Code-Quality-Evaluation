import bisect
from collections import defaultdict

class Solution:
    def solveQueries(self, nums, queries):
        n = len(nums)
        mapping = defaultdict(list)
        for i, num in enumerate(nums):
            mapping[num].append(i)
        
        res = []
        for q in queries:
            x = nums[q]
            lst = mapping[x]
            if len(lst) == 1:
                res.append(-1)
            else:
                idx = bisect.bisect_left(lst, q)
                pred = lst[idx - 1] if idx > 0 else lst[-1]
                if idx == len(lst) - 1:
                    succ = lst[0]
                else:
                    succ = lst[idx + 1]
                best = float('inf')
                for cand in [pred, succ]:
                    d = abs(q - cand)
                    circular_d = min(d, n - d)
                    if circular_d < best:
                        best = circular_d
                res.append(best)
        return res