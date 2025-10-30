class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the total value of coins
        total_value = 75 * x + 10 * y
        
        # Each turn, a player needs to pick coins with a total value of 115
        # We need to determine if the total value can be exactly divided by 115
        # If the total value is not a multiple of 115, the player who starts will lose
        # If the total value is a multiple of 115, the player who starts will win if they play optimally
        
        # Since Alice starts, if the number of turns (total_value // 115) is odd, Alice will lose
        # If the number of turns is even, Bob will lose
        
        turns = total_value // 115
        
        if turns % 2 == 0:
            return "Bob"
        else:
            return "Alice"