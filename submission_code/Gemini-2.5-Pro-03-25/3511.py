class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        """
        Determines the winner of a game where players take turns picking coins.
        Each turn, a player must pick up coins with a total value of 115.
        The available coins have values 75 and 10.
        Alice starts first. The player who cannot make a move loses.

        Args:
            x: The initial number of 75-value coins.
            y: The initial number of 10-value coins.

        Returns:
            "Alice" if Alice wins, "Bob" if Bob wins.
        """
        
        # First, determine the composition of the move required (total value 115).
        # Let 'a' be the number of 75-value coins and 'b' be the number of 10-value coins picked.
        # We need a * 75 + b * 10 = 115, where a and b are non-negative integers.
        # Possible values for 'a':
        # If a = 0: 10b = 115 -> b = 11.5 (not an integer)
        # If a = 1: 75 + 10b = 115 -> 10b = 40 -> b = 4. This is a valid combination (a=1, b=4).
        # If a = 2: 150 + 10b = 115 -> 10b = -35 (not possible for non-negative b)
        # So, the only possible move is to pick exactly 1 coin of value 75 and 4 coins of value 10.
        
        # A player can make a move if and only if they have at least 1 coin of value 75 (x >= 1)
        # and at least 4 coins of value 10 (y >= 4).
        
        # Each move consumes 1 coin of value 75 and 4 coins of value 10.
        # The total number of moves possible throughout the game is limited by the initial counts of coins.
        # The number of times we can pick 1 coin of value 75 is limited by x.
        # The number of times we can pick 4 coins of value 10 is limited by y // 4.
        # The total number of moves possible is the minimum of these two limits.
        
        num_moves = min(x, y // 4)
        
        # Alice and Bob alternate turns. Alice starts.
        # Alice plays turns 1, 3, 5, ...
        # Bob plays turns 2, 4, 6, ...
        
        # The game ends after `num_moves` turns. The player whose turn it is after the last move cannot play and loses.
        # If `num_moves` is odd, the last move is turn `num_moves`, played by Alice. Bob cannot play next. Alice wins.
        # If `num_moves` is even and positive, the last move is turn `num_moves`, played by Bob. Alice cannot play next. Bob wins.
        # If `num_moves` is 0, Alice cannot make the first move. Bob wins immediately.
        
        # We can determine the winner based on the parity of `num_moves`.
        if num_moves % 2 == 1:
            # If the total number of moves is odd, the first player (Alice) wins.
            return "Alice"
        else:
            # If the total number of moves is even (including 0), the second player (Bob) wins.
            return "Bob"