class Solution:
    def canAliceWin(self, n: int) -> bool:
        # rem: stones remaining
        # current: 0 for Alice's turn, 1 for Bob's turn
        rem = n
        current = 0  # Alice starts
        
        # Players must remove 10, 9, 8, ..., 1 stones in order
        for k in range(10, 0, -1):
            # If the current player can't remove k stones, they lose
            if rem < k:
                # current player loses -> return True if opponent is Alice
                return current == 1
            # Remove stones and switch turn
            rem -= k
            current ^= 1
        
        # If all moves 10->1 succeeded, next player has no move and loses
        # current now points to the player who would move next (and can't)
        # So that player loses; return True if opponent is Alice
        return current == 1