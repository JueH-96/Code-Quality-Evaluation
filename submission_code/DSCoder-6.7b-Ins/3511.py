class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        if x % 2 == 0 or y % 2 == 0:
            return "Alice"
        else:
            return "Bob"