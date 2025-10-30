class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        moves = min(x, y // 4)
        return "Alice" if moves % 2 == 1 else "Bob"