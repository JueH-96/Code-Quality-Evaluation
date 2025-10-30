class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        remaining = n - 10
        current_required = 9
        current_player_is_bob = True  # Bob's turn after Alice's first move
        
        while True:
            if remaining < current_required or current_required <= 0:
                return current_player_is_bob
            remaining -= current_required
            current_required -= 1
            current_player_is_bob = not current_player_is_bob