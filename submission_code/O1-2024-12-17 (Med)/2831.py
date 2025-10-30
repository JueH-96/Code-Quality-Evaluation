class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        from math import gcd

        def first_digit(x: int) -> int:
            while x >= 10:
                x //= 10
            return x

        count = 0
        n = len(nums)
        for i in range(n):
            fi = first_digit(nums[i])
            for j in range(i + 1, n):
                lj = nums[j] % 10
                if gcd(fi, lj) == 1:
                    count += 1
        return count