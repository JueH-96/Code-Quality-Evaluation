from math import gcd

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def first_digit(n: int) -> int:
            return ((n - 1) // 10**((n := len(str(n))) - 1)) + 1 if n >= 10 else n

        ans = 0
        for i, num1 in enumerate(nums):
            first = first_digit(num1)
            for num2 in nums[i + 1 :]:
                last = num2 % 10
                if gcd(first, last) == 1:
                    ans += 1
        return ans