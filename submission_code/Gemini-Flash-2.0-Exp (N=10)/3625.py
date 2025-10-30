class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        n -= 10
        turn = 1
        stones_to_remove = 9
        while n >= stones_to_remove:
            n -= stones_to_remove
            stones_to_remove -= 1
            turn += 1
        return turn % 2 == 1