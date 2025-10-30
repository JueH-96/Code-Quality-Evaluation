class Solution:
    def waysToReachStair(self, k: int) -> int:
        ans = 0
        for u in range(0, 62):
            total_up = 1 << u
            d = total_up - k
            if d < 0:
                continue
            if d <= u + 1:
                n = u + 1
                r = min(d, n - d)
                comb = 1
                for i in range(1, r + 1):
                    comb = comb * (n - i + 1) // i
                ans += comb
        return ans