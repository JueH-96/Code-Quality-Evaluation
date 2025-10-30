class Solution:
    def canAliceWin(self, n: int) -> bool:
        stones_to_remove = 10
        total_removed = 0
        is_alice_turn = True
        
        while stones_to_remove > 0:
            # Check if current player can make their move
            if total_removed + stones_to_remove > n:
                # Current player cannot make the move
                return not is_alice_turn  # If Alice can't move, she loses; if Bob can't move, Alice wins
            
            # Make the move
            total_removed += stones_to_remove
            stones_to_remove -= 1
            is_alice_turn = not is_alice_turn
        
        # After all valid moves (10, 9, 8, ..., 1), it would be Alice's turn
        # She would need to remove 0 stones, which is not valid
        return False