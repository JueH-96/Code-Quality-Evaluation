class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the maximum number of moves possible
        # Each move requires at least one 75-coin and four 10-coins
        moves = min(x, y // 4)
        
        # If the number of moves is odd, Alice wins (since she starts)
        if moves % 2 == 1:
            return "Alice"
        else:
            return "Bob"