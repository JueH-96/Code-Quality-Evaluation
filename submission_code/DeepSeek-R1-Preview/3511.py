class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        m = min(x, y // 4)
        if m == 0:
            return "Bob"
        return "Alice" if m % 2 == 1 else "Bob"