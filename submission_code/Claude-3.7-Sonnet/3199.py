class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for a in range(min(n, limit) + 1):
            b_min = max(0, n - a - limit)
            b_max = min(n - a, limit)
            if b_max >= b_min:
                count += (b_max - b_min + 1)
        return count