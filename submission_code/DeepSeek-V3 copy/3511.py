class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the maximum number of moves possible
        # Each move requires at least one 75-coin and four 10-coins
        moves = min(x, y // 4)
        # Determine the winner based on the parity of the number of moves
        if moves % 2 == 1:
            return "Bob"
        else:
            return "Alice"