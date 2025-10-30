class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        
        # Alice starts by removing 10 stones
        n -= 10
        turn = 1  # Bob's turn
        
        while n > 0:
            if n < turn:
                return turn % 2 == 1  # If it's Bob's turn and he can't move, Alice wins
            n -= turn
            turn += 1
        
        return turn % 2 == 0  # If the loop ends and it's Alice's turn, Bob wins