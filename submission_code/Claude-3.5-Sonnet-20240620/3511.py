class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        total_turns = (x * 75 + y * 10) // 115
        
        if total_turns % 2 == 0:
            return "Bob"
        else:
            return "Alice"