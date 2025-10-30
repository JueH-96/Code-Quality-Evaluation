class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        total_moves = min(x, y // 4)
        return "Alice" if total_moves % 2 == 1 else "Bob"