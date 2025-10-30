class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        """
        Determines the winner of the coin game.

        The only way to form a sum of 115 is by using one 75-coin and four 10-coins.
        Thus, each turn consumes 1 coin of value 75 and 4 coins of value 10.

        The total number of turns possible is limited by the minimum of the number of 75-coins
        and the number of sets of four 10-coins.
        """
        
        # Calculate the total number of turns that can be played in the game.
        # A turn requires one 75-coin and four 10-coins.
        # The number of available sets of four 10-coins is y // 4.
        num_turns = min(x, y // 4)
        
        # Alice starts. She plays on turns 1, 3, 5, ...
        # Bob plays on turns 2, 4, 6, ...
        # If the total number of turns is odd, Alice makes the last move and wins.
        # If the total number of turns is even (including 0), Bob makes the last move (or no move is made)
        # and Alice loses on her next turn.
        
        if num_turns % 2 == 1:
            return "Alice"
        else:
            return "Bob"