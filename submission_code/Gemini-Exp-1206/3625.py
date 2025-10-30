class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        
        stones_removed = 10
        remaining_stones = n - stones_removed
        turn = 1
        
        while remaining_stones >= 0:
            if turn % 2 == 1: # Bob's turn
                stones_to_remove = stones_removed - 1
                if stones_to_remove <= 0:
                    return True
                if remaining_stones < stones_to_remove:
                    return True
                remaining_stones -= stones_to_remove
                stones_removed = stones_to_remove
            else: # Alice's turn
                stones_to_remove = stones_removed - 1
                if stones_to_remove <= 0:
                    return False
                if remaining_stones < stones_to_remove:
                    return False
                remaining_stones -= stones_to_remove
                stones_removed = stones_to_remove
            turn += 1
        
        return False