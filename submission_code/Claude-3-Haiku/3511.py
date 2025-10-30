class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the total number of coins available
        total_coins = x * 75 + y * 10
        
        # If the total number of coins is not divisible by 115, Alice will always win
        if total_coins % 115 != 0:
            return "Alice"
        
        # If the total number of coins is divisible by 115, the game will be a tie
        if (x + y) % 2 == 0:
            return "Alice"
        else:
            return "Bob"