class Solution:
    def canAliceWin(self, n: int) -> bool:
        # If n < 10, Alice can't make her first move
        if n < 10:
            return False
        
        stones_left = n - 10  # After Alice's first turn
        stones_to_remove = 9  # Bob's next turn
        alice_turn = False  # Now it's Bob's turn
        
        while stones_left >= stones_to_remove:
            stones_left -= stones_to_remove
            stones_to_remove -= 1
            alice_turn = not alice_turn
        
        # The current player can't make a move, so they lose
        return not alice_turn