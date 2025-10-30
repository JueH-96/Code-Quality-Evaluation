class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each turn requires 1 coin of value 75 and 4 coins of value 10
        # Maximum possible turns is limited by available coins
        max_turns = min(x, y // 4)
        
        # If max_turns is odd, Alice wins (she makes the last move)
        # If max_turns is even, Bob wins (Alice can't move on her turn)
        return "Alice" if max_turns % 2 == 1 else "Bob"