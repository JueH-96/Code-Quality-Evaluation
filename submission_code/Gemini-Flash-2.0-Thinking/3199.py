class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for c1 in range(n + 1):
            if c1 > limit:
                continue
            for c2 in range(n - c1 + 1):
                if c2 > limit:
                    continue
                c3 = n - c1 - c2
                if 0 <= c3 <= limit:
                    count += 1
        return count