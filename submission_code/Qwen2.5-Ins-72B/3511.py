class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the number of turns the game can last
        turns = min(x, y // 4)
        
        # If the number of turns is even, Bob wins; otherwise, Alice wins
        if turns % 2 == 0:
            return "Bob"
        else:
            return "Alice"