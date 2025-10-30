class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        left, right = 1, 10**18
        while left < right:
            mid = (left + right + 1) // 2
            if self.getPrice(mid, x) <= k:
                left = mid
            else:
                right = mid - 1
        return left

    def getPrice(self, num: int, x: int) -> int:
        price = 0
        for i in range(60, -1, -1):
            if (1 << i) > num:
                continue
            if i % x == 0:
                price += num // (1 << i)
            num %= 1 << i
        return price