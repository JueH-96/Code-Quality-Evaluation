class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        target = 115
        total = x * 75 + y * 10

        if total % target != 0:
            return "Alice"  # as Alice moves first, she can move optimally and take advantage of the remaining
        if total // target % 2 == 0:
            return "Bob"
        return "Alice"