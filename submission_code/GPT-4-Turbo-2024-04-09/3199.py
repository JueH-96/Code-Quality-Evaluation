class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for x in range(min(n, limit) + 1):
            for y in range(min(n - x, limit) + 1):
                z = n - x - y
                if z <= limit:
                    count += 1
        return count