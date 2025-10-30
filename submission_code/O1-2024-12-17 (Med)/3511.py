class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each move requires exactly 1 coin of value 75 and 4 coins of value 10.
        # Thus, the number of moves possible is limited by:
        #   1) The number of 75-value coins (x).
        #   2) The integer division of the number of 10-value coins by 4 (y // 4).
        # Let M = min(x, y // 4).
        #
        # Players alternate turns, starting with Alice.
        # If M is odd, the last (M-th) turn belongs to Alice (she makes the last valid move),
        # so Bob cannot move next and loses, meaning Alice wins.
        # If M is even, Bob makes the last valid move and Alice cannot move next, so Bob wins.
        
        M = min(x, y // 4)
        return "Alice" if M % 2 == 1 else "Bob"