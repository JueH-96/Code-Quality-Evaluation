class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each player needs to pick coins with a total value of 115
        # We need to check how many turns can be made with the given coins
        
        # Calculate the maximum number of turns possible
        turns = min(x, y // 4)
        
        # After the maximum number of turns, check the remaining coins
        remaining_x = x - turns
        remaining_y = y - turns * 4
        
        # If there are no remaining coins to make another turn, the last player to play wins
        if remaining_x == 0 or remaining_y < 4:
            return "Alice" if turns % 2 == 0 else "Bob"
        else:
            return "Bob" if turns % 2 == 0 else "Alice"