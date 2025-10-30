class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for i in range(min(n, limit) + 1):
            for j in range(min(n - i, limit) + 1):
                k = n - i - j
                if k <= limit and k >= 0:
                    count += 1
        return count