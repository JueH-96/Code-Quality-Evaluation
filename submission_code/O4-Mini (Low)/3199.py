class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        # brute‐force over x and y, compute z = n - x - y
        # and check 0 <= z <= limit
        for x in range(limit + 1):
            for y in range(limit + 1):
                z = n - x - y
                if 0 <= z <= limit:
                    count += 1
        return count