class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        """
        Determines the winner of the game where players pick up coins with a total value of 115.
        """
        num_moves = min(x, y // 4)

        # If the number of possible moves is odd, Alice wins.
        # If the number of possible moves is even, Bob wins.

        if num_moves % 2 == 1:
            return "Alice"
        else:
            return "Bob"