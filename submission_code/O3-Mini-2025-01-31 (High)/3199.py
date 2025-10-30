class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        ways = 0
        for a in range(limit + 1):
            for b in range(limit + 1):
                c = n - a - b
                if 0 <= c <= limit:
                    ways += 1
        return ways