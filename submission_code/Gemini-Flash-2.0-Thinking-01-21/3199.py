class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for x in range(limit + 1):
            start_y = max(0, n - limit - x)
            end_y = min(limit, n - x)
            if start_y <= end_y:
                count += (end_y - start_y + 1)
        return count