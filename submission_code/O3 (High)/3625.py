class Solution:
    def canAliceWin(self, n: int) -> bool:
        """
        Simulate the game.
        turn : 0 for Alice, 1 for Bob, 2 for Alice, ...
        required : stones the current player must take
        """
        required = 10          # stones Alice must take on the first turn
        turn = 0               # 0 = Alice, 1 = Bob, …
        
        while True:
            # If the current player cannot take the required number of stones, that player loses
            if n < required:
                # If it's Alice's turn (even turn index) and she can't move, Alice loses -> False
                # If it's Bob's turn (odd turn index) and he can't move, Alice wins -> True
                return turn % 2 == 1     # True when Bob loses
            
            # Otherwise, current player makes the move
            n -= required
            turn += 1
            required -= 1      # each subsequent move requires 1 fewer stone
            
            # Once required reaches 0 (after 10 successful moves), the next
            # player would have to take 0 stones and therefore cannot move.
            if required == 0:
                # Turn has already been incremented; it's now the next player's turn.
                # If it's Alice's turn and she must take 0, she loses; otherwise Bob loses.
                return turn % 2 == 1