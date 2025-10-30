class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        for i in range(1, limit+1):
            for j in range(i, limit+1):
                for k in range(j, limit+1):
                    if i + j + k <= n:
                        count += 1
        return count