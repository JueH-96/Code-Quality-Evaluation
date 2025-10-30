class Solution:
    def canAliceWin(self, n: int) -> bool:
        stones_to_remove = 10
        alice_turn = True
        
        while n > 0:
            if n < stones_to_remove:
                # Current player cannot make a move, so they lose
                return not alice_turn
            
            n -= stones_to_remove
            stones_to_remove -= 1
            alice_turn = not alice_turn
        
        # If we get here, someone just took the last stone(s)
        # The player who just moved wins
        return alice_turn