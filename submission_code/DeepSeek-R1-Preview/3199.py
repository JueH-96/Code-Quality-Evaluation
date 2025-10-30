class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        max_a = min(n, limit)
        for a in range(0, max_a + 1):
            remaining = n - a
            max_b = min(remaining, limit)
            for b in range(0, max_b + 1):
                c = remaining - b
                if c <= limit:
                    count += 1
        return count