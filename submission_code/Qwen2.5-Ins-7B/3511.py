class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        if (x * 75 + y * 10) % 115 != 0:
            return "Bob"
        if x * 3 + y * 2 >= 115:
            return "Bob"
        return "Alice"