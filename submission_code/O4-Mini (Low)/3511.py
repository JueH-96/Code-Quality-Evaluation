class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        """
        Each move consists of removing exactly one 75‐coin and four 10‐coins,
        since 75*1 + 10*4 = 115 is the only way to sum to 115 with these coin values.
        The total number of moves possible is t = min(x, y // 4).
        If t is odd, Alice (who starts) makes the last valid move and wins;
        otherwise Bob wins.
        """
        t = min(x, y // 4)
        return "Alice" if t % 2 == 1 else "Bob"