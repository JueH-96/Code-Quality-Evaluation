class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the maximum number of turns possible
        turns = min(x, y // 4)
        
        # Determine the winner based on whether the number of turns is odd or even
        if turns % 2 == 1:  # Odd number of turns
            return "Alice"
        else:  # Even number of turns
            return "Bob"