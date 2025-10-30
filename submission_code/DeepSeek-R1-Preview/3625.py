class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        n -= 10
        current_required = 9
        is_bob_turn = True
        
        while True:
            if is_bob_turn:
                if current_required <= 0 or n < current_required:
                    return True
                n -= current_required
                current_required -= 1
                is_bob_turn = False
            else:
                if current_required <= 0 or n < current_required:
                    return False
                n -= current_required
                current_required -= 1
                is_bob_turn = True