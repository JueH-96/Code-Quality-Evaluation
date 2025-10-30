from typing import List
import math

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        # Precompute first digit of each number and last digit of each number.
        n = len(nums)
        first_digits = []
        last_digits = []
        
        for num in nums:
            s = str(num)
            first_digits.append(int(s[0]))
            last_digits.append(num % 10)  # last digit
        
        count = 0
        # Iterate through every pair (i, j) such that i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Check if first digit of nums[i] and last digit of nums[j] are coprime.
                if math.gcd(first_digits[i], last_digits[j]) == 1:
                    count += 1
        return count