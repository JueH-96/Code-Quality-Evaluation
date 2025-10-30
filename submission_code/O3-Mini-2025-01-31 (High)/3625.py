class Solution:
    def canAliceWin(self, n: int) -> bool:
        # turn_is_alice is True when it is Alice's turn, False when it is Bob's.
        turn_is_alice = True
        move = 10  # The first move requires removing 10 stones.
        
        # Continue making moves as long as the player can remove the exact required number of stones.
        # The required number decreases by exactly 1 after each move.
        while move > 0 and n >= move:
            n -= move
            move -= 1  # The next move is exactly 1 stone less.
            turn_is_alice = not turn_is_alice  # Switch turns.
        
        # When the loop exits, the current player (whose turn it is) cannot make a move.
        # Therefore, that player loses.
        # If it's Alice's turn, she loses (so we return False); if it's Bob's turn, Alice wins (return True).
        return not turn_is_alice

# Example usage:
# sol = Solution()
# print(sol.canAliceWin(12))  # Should output: True
# print(sol.canAliceWin(1))   # Should output: False