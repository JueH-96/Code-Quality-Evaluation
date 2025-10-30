class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        if x > y:
            return "Alice"
        elif x < y:
            return "Bob"
        else:
            return "Tie"