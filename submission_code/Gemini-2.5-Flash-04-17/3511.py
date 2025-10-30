class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # The only combination of coins that sums to 115 is 1 coin of 75 and 4 coins of 10.
        # This is derived from the equation 75a + 10b = 115 for non-negative integers a, b.
        # Dividing by 5 gives 15a + 2b = 23.
        # If a=0, 2b=23 (no integer solution).
        # If a=1, 15 + 2b = 23 => 2b = 8 => b = 4 (valid solution: 1 coin of 75, 4 coins of 10).
        # If a>=2, 15a >= 30 > 23, so no non-negative b is possible.
        # Thus, each turn requires exactly 1 coin of value 75 and 4 coins of value 10.

        # The maximum number of turns possible is limited by the available coins for each type.
        # The number of times we can take 1 coin of 75 is x // 1 = x.
        # The number of times we can take 4 coins of 10 is y // 4.

        # A turn can only be completed if both requirements are met.
        # So, the total number of turns possible is the minimum of these two counts.
        max_turns = min(x, y // 4)

        # Alice goes first.
        # The game is finite, deterministic, with perfect information, and no ties.
        # The player unable to move loses.
        # This is a standard impartial game structure. The winner is determined by the total number of moves possible.
        # If the total number of moves (max_turns) is odd, the first player (Alice) wins.
        # If the total number of moves (max_turns) is even, the second player (Bob) wins.

        if max_turns % 2 == 1:
            # If the total number of possible turns is odd, Alice wins.
            return "Alice"
        else:
            # If the total number of possible turns is even, Bob wins.
            return "Bob"