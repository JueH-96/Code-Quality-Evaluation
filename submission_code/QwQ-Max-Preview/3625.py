class Solution:
    def canAliceWin(self, n: int) -> bool:
        current_stones = n
        current_required = 10
        is_alice_turn = True
        
        while True:
            # Check if current_required is invalid (<=0)
            if current_required < 1:
                return not is_alice_turn
            
            # Check if current player can take the required stones
            if current_stones < current_required:
                return not is_alice_turn
            
            # Take the stones
            current_stones -= current_required
            is_alice_turn = not is_alice_turn
            current_required -= 1