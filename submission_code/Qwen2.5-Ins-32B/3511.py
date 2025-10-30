class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the number of complete turns
        turns = min(x, y // 4)
        
        # If the number of turns is even, Bob wins; otherwise, Alice wins
        return "Bob" if turns % 2 == 0 else "Alice"