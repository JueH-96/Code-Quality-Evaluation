class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        ans = 0
        for i in range(0, min(n, limit) + 1):
            lower = max(0, n - limit - i)
            upper = min(limit, n - i)
            if lower <= upper:
                ans += upper - lower + 1
        return ans