class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each turn, a player must pick exactly 1 coin of value 75 and 4 coins of value 10.
        # The total number of such picks possible is limited by:
        #   1) having at least one 75-value coin
        #   2) having at least four 10-value coins
        # Let n = min(x, y // 4). This represents the maximum number of turns in the game.
        #
        # The turns alternate: Alice, Bob, Alice, Bob, ...
        # If n is odd, Alice takes the last valid turn, so Bob faces a turn with no move and loses → Alice wins.
        # If n is even, Bob takes the last valid turn, so Alice faces a turn with no move and loses → Bob wins.
        n = min(x, y // 4)
        
        # If n is even → Bob wins, else Alice wins.
        return "Bob" if n % 2 == 0 else "Alice"