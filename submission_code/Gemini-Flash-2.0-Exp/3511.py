class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        turns = 0
        while x >= 0 and y >= 0:
            if x > 0 and y >= 4:
                x -= 1
                y -= 4
                turns += 1
            elif x == 0 and y >= 12:
                y -= 12
                turns += 1
            else:
                break
        if turns % 2 == 0:
            return "Bob"
        else:
            return "Alice"