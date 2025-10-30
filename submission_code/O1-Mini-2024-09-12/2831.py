from math import gcd
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def get_first_digit(num):
            while num >= 10:
                num //= 10
            return num

        count = 0
        n = len(nums)
        for i in range(n):
            first_digit = get_first_digit(nums[i])
            for j in range(i + 1, n):
                last_digit = nums[j] % 10
                if gcd(first_digit, last_digit) == 1:
                    count += 1
        return count