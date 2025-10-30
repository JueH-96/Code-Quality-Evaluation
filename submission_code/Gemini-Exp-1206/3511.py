class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        turns = 0
        while x > 0 and y >= 4:
            x -= 1
            y -= 4
            turns += 1
        if turns % 2 == 0:
            return "Alice"
        else:
            return "Bob"