class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        k = min(x, y // 4)
        if k == 0:
            return "Bob"
        return "Alice" if k % 2 == 1 else "Bob"