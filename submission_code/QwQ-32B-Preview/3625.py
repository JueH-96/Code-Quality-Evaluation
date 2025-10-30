class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False  # Alice cannot make the first move
        
        def can_win(stones, alice_turn, required_removal):
            if stones < required_removal:
                # Current player cannot make a move, so the previous player wins
                return not alice_turn
            else:
                next_removal = required_removal - 1
                if next_removal <= 0:
                    # Current player cannot make a move, so they lose
                    return alice_turn
                else:
                    return can_win(stones - required_removal, not alice_turn, next_removal)
        
        # Alice starts by removing 10 stones
        return can_win(n, True, 10)