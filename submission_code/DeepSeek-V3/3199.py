class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for a in range(min(n, limit) + 1):
            for b in range(min(n - a, limit) + 1):
                c = n - a - b
                if c <= limit:
                    count += 1
        return count