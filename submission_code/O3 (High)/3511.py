class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        """
        Every valid move must consist of
            1 coin of value 75  and  4 coins of value 10  (75 + 4*10 = 115)

        Therefore, the total number of complete turns that can be played is
            turns = min( number_of_75_coins,
                         number_of_10_coins // 4 )

        • If turns is 0, Alice has no legal move and immediately loses → Bob wins.
        • After each complete turn the right to move alternates.
          Hence, if turns is odd Alice made the last valid move and Bob is stuck,
          otherwise Bob made the last valid move and Alice is stuck.

        The function returns the player who wins with optimal play.
        """
        turns = min(x, y // 4)          # maximal number of full 115-valued moves

        # Winner is the player who made the last valid move
        return "Alice" if turns % 2 == 1 else "Bob"