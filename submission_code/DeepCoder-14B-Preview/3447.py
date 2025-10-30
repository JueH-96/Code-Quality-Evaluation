class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        n = min(x, y // 4)
        return "Alice" if n % 2 == 1 else "Bob"