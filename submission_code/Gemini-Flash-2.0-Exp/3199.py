class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for i in range(min(n + 1, limit + 1)):
            for j in range(min(n - i + 1, limit + 1)):
                k = n - i - j
                if k <= limit:
                    count += 1
        return count