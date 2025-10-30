class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        from math import gcd

        def first_digit(n):
            while n >= 10:
                n //= 10
            return n

        def last_digit(n):
            return n % 10

        count = 0
        n = len(nums)

        for i in range(n):
            first = first_digit(nums[i])
            for j in range(i + 1, n):
                last = last_digit(nums[j])
                if gcd(first, last) == 1:
                    count += 1

        return count