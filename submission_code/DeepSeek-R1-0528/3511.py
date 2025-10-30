class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        moves = min(x, y // 4)
        return "Bob" if moves % 2 == 0 else "Alice"