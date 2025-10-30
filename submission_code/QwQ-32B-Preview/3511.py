class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the number of possible moves
        m = min(x, y // 4)
        # Determine the winner based on the parity of m
        if m % 2 == 1:
            return "Alice"
        else:
            return "Bob"