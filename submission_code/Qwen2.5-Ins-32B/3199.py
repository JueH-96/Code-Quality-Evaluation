class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for i in range(max(0, n - 2 * limit), min(n, limit) + 1):
            for j in range(max(0, n - i - limit), min(n - i, limit) + 1):
                k = n - i - j
                if 0 <= k <= limit:
                    count += 1
        return count