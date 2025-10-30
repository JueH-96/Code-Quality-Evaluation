class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the maximum number of turns possible
        max_turns = min(x, y // 4)
        
        # Calculate remaining coins after max_turns
        remaining_x = x - max_turns
        remaining_y = y - max_turns * 4
        
        # If there are no remaining coins for another turn, the player who cannot play loses
        if remaining_x == 0 and remaining_y < 4:
            return "Bob" if max_turns % 2 == 1 else "Alice"
        elif remaining_x < 0 or remaining_y < 0:
            return "Bob" if max_turns % 2 == 1 else "Alice"
        
        # If there are remaining coins, the player who cannot play loses
        return "Alice" if max_turns % 2 == 1 else "Bob"