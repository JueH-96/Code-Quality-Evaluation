import math
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                first = int(str(nums[i])[0])
                last = nums[j] % 10
                if math.gcd(first, last) == 1:
                    count += 1
        return count