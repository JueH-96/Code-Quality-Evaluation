class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for x1 in range(min(n, limit) + 1):
            for x2 in range(min(n - x1, limit) + 1):
                x3 = n - x1 - x2
                if 0 <= x3 <= limit:
                    count += 1
        return count