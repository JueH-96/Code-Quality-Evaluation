class Solution:
    def canAliceWin(self, n: int) -> bool:
        # Alice starts by removing 10 stones
        n -= 10
        
        # If there are no stones left, Alice loses
        if n <= 0:
            return False
        
        # For each subsequent turn, the player removes 1 fewer stone than the previous opponent
        while n > 0:
            n -= 11
            if n <= 0:
                return True
            n -= 10
            if n <= 0:
                return False
        
        # If the loop completes without a winner, return False (Bob wins)
        return False