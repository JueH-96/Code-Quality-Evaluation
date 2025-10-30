from math import gcd
from typing import List

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                # Get the first digit of nums[i] and the last digit of nums[j]
                first_digit = int(str(nums[i])[0])
                last_digit = nums[j] % 10
                
                # Check if the first digit and the last digit are coprime
                if gcd(first_digit, last_digit) == 1:
                    count += 1
        return count