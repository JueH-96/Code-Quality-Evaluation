class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        
        stones_removed = 10
        while n >= stones_removed:
            n -= stones_removed
            stones_removed -= 1
        
        return True