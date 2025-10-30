class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        if n <= 18:
            return True
        if n <= 26:
            return False
        if n <= 33:
            return True
        if n <= 39:
            return False
        if n <= 44:
            return True
        if n <= 48:
            return False
        return True