class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        max_turns = min(x, y // 4)
        if max_turns % 2 == 1:
            return "Alice"
        else:
            return "Bob"