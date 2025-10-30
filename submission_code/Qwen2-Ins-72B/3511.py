class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the total number of moves possible
        total_moves = (x * 75 + y * 10) // 115
        
        # Determine the winner based on the parity of total_moves
        if total_moves % 2 == 0:
            return "Bob"
        else:
            return "Alice"