class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        remaining = n - 10
        if remaining < 9:
            return True
        return False