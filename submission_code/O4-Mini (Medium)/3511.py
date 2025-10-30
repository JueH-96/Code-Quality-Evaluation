class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each move consumes exactly 1 coin of value 75 and 4 coins of value 10.
        # The total number of moves possible is limited by x and by y//4.
        moves = min(x, y // 4)
        # If the number of moves is odd, Alice (who starts) makes the last move and wins.
        # If it's even (including zero), Bob wins.
        return "Alice" if moves % 2 == 1 else "Bob"