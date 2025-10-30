class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        stones = n
        turn = 1
        alice_turn = True
        stones_removed = 10
        while stones > 0:
            if alice_turn:
                if stones >= stones_removed:
                    stones -= stones_removed
                else:
                    return False
            else:
                if stones >= stones_removed:
                    stones -= stones_removed
                else:
                    return True
            
            if turn %2 == 0:
                stones_removed = (turn //2) + 10 - (turn //2) *2
            else:
                stones_removed = (turn //2) + 10 - (turn //2) *2
                
            if stones_removed < 1:
                stones_removed = 1
            
            alice_turn = not alice_turn
            turn += 1
        return not alice_turn