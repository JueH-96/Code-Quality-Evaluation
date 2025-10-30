class Solution:
    def stoneGame(self, x: int, y: int) -> str:
        # Calculate the number of turns
        turns = min(x, (y // 4))
        
        # If the total number of turns is even, Bob wins; otherwise, Alice wins
        if turns % 2 == 0:
            return "Bob"
        else:
            return "Alice"