class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # If the total value of coins is a multiple of 115, the player who cannot make a move loses.
        total_value = 75 * x + 10 * y

        # If the total value is not a multiple of 115, Alice wins because she goes first.
        if total_value % 115 != 0:
            return "Alice"

        # If the total value is a multiple of 115, we need to check the number of moves.
        # If the number of moves is odd, Alice wins because she goes first.
        # If the number of moves is even, Bob wins because he goes second.
        num_moves = total_value // 115
        if num_moves % 2 == 1:
            return "Alice"
        else:
            return "Bob"