class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Analyze the coin requirements for one turn:
        # A player needs to pick coins with a total value of 115.
        # Available coin values: 75 and 10.
        # Let n_75 be the count of 75-value coins and n_10 be the count of 10-value coins.
        # Equation: 75 * n_75 + 10 * n_10 = 115
        
        # By trying possible integer values for n_75:
        # If n_75 = 0: 10 * n_10 = 115  => n_10 = 11.5 (not an integer)
        # If n_75 = 1: 75 * 1 + 10 * n_10 = 115 => 10 * n_10 = 40 => n_10 = 4 (valid combination)
        # If n_75 = 2: 75 * 2 + 10 * n_10 = 115 => 150 + 10 * n_10 = 115 => 10 * n_10 = -35 (not possible)
        
        # Thus, the only way to pick coins summing to 115 is to pick 1 coin of value 75
        # and 4 coins of value 10.
        
        # Calculate how many turns are possible based on the availability of 'x' coins (value 75).
        # Each turn consumes 1 'x' coin.
        possible_turns_from_x = x // 1 # This is simply x

        # Calculate how many turns are possible based on the availability of 'y' coins (value 10).
        # Each turn consumes 4 'y' coins.
        possible_turns_from_y = y // 4

        # The game can only proceed for as many turns as allowed by the most limiting resource.
        # So, the total number of complete turns that can be made is the minimum of these two.
        total_playable_turns = min(possible_turns_from_x, possible_turns_from_y)

        # Alice starts the game.
        # If the total number of turns is odd, Alice makes the last move and wins.
        # If the total number of turns is even (including 0), Bob makes the last move (or Alice can't make the first move), and Bob wins.
        if total_playable_turns % 2 == 1:
            return "Alice"
        else:
            return "Bob"