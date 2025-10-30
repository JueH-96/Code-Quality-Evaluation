class Solution:
    def canAliceWin(self, n: int) -> bool:
        """
        Determines if Alice can win the stone game based on pre-calculated winning ranges.
        
        The game is deterministic. The winner depends solely on the initial number of stones, n.
        Alice wins if Bob is the player who cannot make a move. By analyzing the game's
        progression, we can identify the specific ranges of n for which this occurs.
        """

        # Alice wins if n falls into one of these ranges:
        # Range 1: 10 <= n <= 18
        # Range 2: 27 <= n <= 33
        # Range 3: 40 <= n <= 44
        # Range 4: 49 <= n <= 50
        # If n is not in these ranges, Alice loses.
        
        if (10 <= n <= 18) or \
           (27 <= n <= 33) or \
           (40 <= n <= 44) or \
           (49 <= n <= 50):
            return True
        
        return False