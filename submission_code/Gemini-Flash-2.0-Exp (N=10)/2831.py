import math
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                first_digit_i = int(str(nums[i])[0])
                last_digit_j = nums[j] % 10
                if math.gcd(first_digit_i, last_digit_j) == 1:
                    count += 1
        return count