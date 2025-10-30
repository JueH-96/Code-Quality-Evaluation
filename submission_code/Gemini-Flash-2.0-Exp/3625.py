class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        
        n -= 10
        
        if n < 9:
            return True
        
        n -= 9
        
        if n < 8:
            return False
        
        n -= 8
        
        if n < 7:
            return True
        
        n -= 7
        
        if n < 6:
            return False
        
        n -= 6
        
        if n < 5:
            return True
        
        n -= 5
        
        if n < 4:
            return False
        
        n -= 4
        
        if n < 3:
            return True
        
        n -= 3
        
        if n < 2:
            return False
        
        n -= 2
        
        if n < 1:
            return True
        
        n -= 1
        
        return False