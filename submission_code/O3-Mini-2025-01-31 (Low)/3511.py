class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # To make a move, one coin of value 75 and four coins of value 10 are required.
        # So the number of moves possible is limited by:
        moves_possible = min(x, y // 4)
        
        # The game is played alternately starting with Alice.
        # If moves_possible is odd, then the last move was made by Alice and Bob loses.
        # If moves_possible is even, then the last move was made by Bob and Alice loses.
        if moves_possible % 2 == 1:
            return "Alice"
        else:
            return "Bob"