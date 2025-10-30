class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        return max(0, min(n, m) - 1)