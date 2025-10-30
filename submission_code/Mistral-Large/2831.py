from typing import List
import math

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def first_digit(n):
            while n >= 10:
                n //= 10
            return n

        def last_digit(n):
            return n % 10

        def are_coprime(a, b):
            return math.gcd(a, b) == 1

        count = 0
        n = len(nums)

        for i in range(n):
            for j in range(i + 1, n):
                first_dig = first_digit(nums[i])
                last_dig = last_digit(nums[j])
                if are_coprime(first_dig, last_dig):
                    count += 1

        return count