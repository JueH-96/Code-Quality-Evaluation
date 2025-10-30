from typing import List
from heapq import *

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        neg = []
        pos = []
        zero = False
        for num in nums:
            if num < 0:
                heappush(neg, num)
            elif num > 0:
                heappush(pos, -num)
            else:
                zero = True
        if len(neg) % 2 == 0:
            neg.sort()
        else:
            neg.sort()
            neg.pop()
        if len(pos) % 2 == 0:
            pos.sort()
        else:
            pos.sort()
            pos.pop()
        strength = 1
        while neg and k > 1:
            strength *= heappop(neg)
            k -= 1
        while pos and k > 1:
            strength *= -heappop(pos)
            k -= 1
        if neg and k == 1:
            strength *= heappop(neg)
            k -= 1
        if pos and k == 1:
            strength *= -heappop(pos)
            k -= 1
        if zero and k % 2 == 1:
            strength = 0
        return strength