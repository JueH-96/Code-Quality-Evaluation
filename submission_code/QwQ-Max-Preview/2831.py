import math
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            first = nums[i]
            while first >= 10:
                first //= 10
            for j in range(i + 1, n):
                last = nums[j] % 10
                if math.gcd(first, last) == 1:
                    count += 1
        return count