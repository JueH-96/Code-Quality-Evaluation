class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each turn requires 1 coin of 75 and 4 coins of 10 to make 115
        # Check how many complete turns are possible
        max_turns = min(x, y // 4)
        
        # If max turns is even, Bob wins since Alice goes first
        # If max turns is odd, Alice wins
        if max_turns % 2 == 0:
            return "Bob"
        return "Alice"