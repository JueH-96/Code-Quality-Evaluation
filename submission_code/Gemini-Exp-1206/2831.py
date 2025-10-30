class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)

        def first_digit(n):
            while n >= 10:
                n //= 10
            return n

        def last_digit(n):
            return n % 10

        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if gcd(first_digit(nums[i]), last_digit(nums[j])) == 1:
                    count += 1
        return count