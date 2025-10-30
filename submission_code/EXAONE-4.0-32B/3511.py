class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        moves = min(x, y // 4)
        if moves % 2 == 0:
            return "Bob"
        else:
            return "Alice"