class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        if n % 2 == 0 or m % 2 == 0:
            return (n * m) // 2
        else:
            return (n * m - 1) // 2