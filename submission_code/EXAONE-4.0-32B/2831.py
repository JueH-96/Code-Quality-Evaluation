import math
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        n = len(nums)
        firsts = []
        for num in nums:
            x = num
            while x >= 10:
                x //= 10
            firsts.append(x)
        lasts = [num % 10 for num in nums]
        
        freq = [0] * 10
        count = 0
        for j in range(n):
            last_j = lasts[j]
            for a in range(1, 10):
                if math.gcd(a, last_j) == 1:
                    count += freq[a]
            first_j = firsts[j]
            freq[first_j] += 1
        
        return count