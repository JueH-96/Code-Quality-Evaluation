class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        max_a = min(limit, n)
        for a in range(max_a + 1):
            remaining = n - a
            max_b = min(limit, remaining)
            for b in range(max_b + 1):
                c = remaining - b
                if c <= limit and c >= 0:
                    count += 1
        return count