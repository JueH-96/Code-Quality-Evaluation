class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        turns = min(x, y // 4)
        if turns % 2 == 1:
            return "Alice"
        else:
            return "Bob"