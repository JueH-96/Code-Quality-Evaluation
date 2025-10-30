class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        N = min(x, y // 4)
        if N % 2 == 1:
            return "Alice"
        else:
            return "Bob"