class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the total number of turns possible
        turns = min(x, y // 4)
        
        # If the number of turns is odd, Alice wins, otherwise Bob wins
        return "Alice" if turns % 2 == 1 else "Bob"