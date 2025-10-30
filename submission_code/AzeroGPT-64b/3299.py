from math import log2, floor, sqrt
from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        if len(nums) == 1: return 1
        c = Counter(nums)
        candidates = [x for x in c.keys() if int(sqrt(x)) == sqrt(x) and 1 < x]
        sequences = []
        for x in candidates:
            cnt = 0
            while x in c:
                cnt += 1
                x *= x
            if cnt > 0:
                sequences.append([cnt, x])
        best = 0
        for seq in sequences:
            cnt, last_pow = seq
            if last_pow == 4:
                ans = 2 * cnt - 1
            else:
                prec = int(log2(last_pow)**(1/2))
                ans = cnt * 2 - prec -1
            best = max(ans, best)
        return best if best > 0 else 1