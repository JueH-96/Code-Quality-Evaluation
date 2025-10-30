class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the number of turns possible
        turns = min(x, y // 4)
        
        # If the number of turns is odd, Alice will be the last to play and win, so Bob loses
        # If the number of turns is even, Bob will be the last to play and win, so Alice loses
        if turns % 2 == 1:
            return "Bob"
        else:
            return "Alice"