class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Each move must take exactly one 75-coin and four 10-coins (75 + 4*10 = 115).
        # The total number of moves possible is min(x, y // 4).
        # If that count is odd, Alice makes the last move and wins; otherwise Bob wins.
        return "Alice" if (min(x, y // 4) % 2 == 1) else "Bob"