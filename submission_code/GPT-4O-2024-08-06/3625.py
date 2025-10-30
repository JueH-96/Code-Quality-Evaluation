class Solution:
    def canAliceWin(self, n: int) -> bool:
        # Alice starts by removing 10 stones
        if n < 10:
            return False
        
        # Calculate the number of stones left after Alice's first move
        stones_left = n - 10
        
        # Determine if Alice can win with the remaining stones
        # Alice wins if the remaining stones are not a multiple of 9
        # because Bob will always leave a multiple of 9 stones for Alice
        return stones_left % 9 != 0