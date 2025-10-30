class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for x in range(0, min(n, limit) + 1):
            for y in range(0, min(n - x, limit) + 1):
                z = n - x - y
                if 0 <= z <= limit:
                    count += 1
        return count