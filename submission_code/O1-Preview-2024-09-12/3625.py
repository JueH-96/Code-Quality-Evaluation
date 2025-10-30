class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        else:
            return n % 19 >= 10