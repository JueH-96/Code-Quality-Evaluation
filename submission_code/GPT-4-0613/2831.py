from typing import List
from math import gcd

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                first_digit = int(str(nums[i])[0])
                last_digit = int(str(nums[j])[-1])
                if gcd(first_digit, last_digit) == 1:
                    count += 1
        return count