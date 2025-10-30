class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        n -= 10
        current_move = 9
        current_turn = True  # True represents Bob's turn
        
        while current_move > 0:
            if n < current_move:
                # Current player cannot take the required stones
                if current_turn:
                    # Bob's turn, he can't take, Alice wins
                    return True
                else:
                    # Alice's turn, she can't take, Bob wins
                    return False
            n -= current_move
            current_move -= 1
            current_turn = not current_turn
        
        # After the loop, current_move is <= 0
        # Check whose turn it is
        if current_turn:
            # Bob's turn, he can't take any stones, Alice wins
            return True
        else:
            # Alice's turn, she can't take any stones, Bob wins
            return False