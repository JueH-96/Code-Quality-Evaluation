from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        ans = 0
        if 1 in freq:
            ans = max(ans, freq[1] if freq[1] % 2 == 1 else freq[1] - 1)
        
        for x in freq:
            if x == 1:
                continue
            chain = []
            cur = x
            while True:
                chain.append(cur)
                if cur * cur > 10**9:
                    break
                nxt = cur * cur
                if nxt not in freq:
                    break
                cur = nxt
            n = len(chain)
            for j in range(n-1, -1, -1):
                if freq[chain[j]] < 1:
                    continue
                valid = True
                for i in range(j):
                    if freq[chain[i]] < 2:
                        valid = False
                        break
                if valid:
                    ans = max(ans, 2 * j + 1)
                    break
        return ans