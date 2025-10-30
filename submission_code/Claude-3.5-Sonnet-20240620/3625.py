class Solution:
    def canAliceWin(self, n: int) -> bool:
        def can_win(stones, turn):
            if stones < turn:
                return False
            if stones == turn:
                return True
            return not can_win(stones - turn, turn - 1)
        
        return can_win(n, 10)