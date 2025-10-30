class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        mod = 10 ** 9 + 7
        if n * (n + 1) < target * 2:
            return n * (n + 1) // 2 % mod
        l = 1
        r = target * 2
        while l < r:
            mid = (l + r) // 2
            if mid * (mid + 1) < target * 2 * (n + 1):
                l = mid + 1
            else:
                r = mid
        return ((l - 1) * l // 2 + (n - (l - 1) // 2) * (l + 1)) % mod