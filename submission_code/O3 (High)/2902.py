from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # two largest numbers for every possible max digit (0-9)
        first = [-1] * 10   # largest
        second = [-1] * 10  # second largest

        # helper to obtain the maximum digit of a number
        def max_digit(x: int) -> int:
            md = 0
            while x:
                md = max(md, x % 10)
                x //= 10
            return md

        for num in nums:
            d = max_digit(num)
            if num > first[d]:
                second[d] = first[d]
                first[d] = num
            elif num > second[d]:
                second[d] = num

        ans = -1
        for d in range(10):
            if second[d] != -1:            # we have at least two numbers with this max digit
                ans = max(ans, first[d] + second[d])

        return ans