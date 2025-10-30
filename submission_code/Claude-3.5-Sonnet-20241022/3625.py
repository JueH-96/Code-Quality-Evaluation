class Solution:
    def canAliceWin(self, n: int) -> bool:
        # If Alice can't make her first move (10 stones), she loses
        if n < 10:
            return False
            
        # After Alice's first move, remaining stones
        remaining = n - 10
        
        # Bob's turn, he needs to remove 9 stones
        stones_to_remove = 9
        
        # Keep alternating turns until someone can't make a move
        while remaining >= 0:
            # Bob's turn
            if remaining < stones_to_remove:
                return True  # Bob can't make a move, Alice wins
            remaining -= stones_to_remove
            stones_to_remove -= 1
            
            # Alice's turn
            if remaining < stones_to_remove:
                return False  # Alice can't make a move, Bob wins
            remaining -= stones_to_remove
            stones_to_remove -= 1
            
        return False  # If we get here, no one can make a move