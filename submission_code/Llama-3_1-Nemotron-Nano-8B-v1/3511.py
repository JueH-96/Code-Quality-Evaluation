class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        k = min(x, y // 4)
        return "Alice" if k % 2 else "Bob"