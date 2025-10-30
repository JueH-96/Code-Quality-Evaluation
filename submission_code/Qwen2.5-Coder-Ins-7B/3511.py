class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        if (x + y) * 10 < 115:
            return "Bob"
        if (x + y) * 75 > 115:
            return "Alice"
        if (x + y) * 10 == 115:
            return "Bob"
        if (x + y) * 75 == 115:
            return "Alice"
        if (x + y) * 10 > 115 and (x + y) * 75 < 115:
            return "Alice"
        if (x + y) * 10 < 115 and (x + y) * 75 > 115:
            return "Bob"
        if (x + y) * 10 == 115 and (x + y) * 75 == 115:
            return "Bob"