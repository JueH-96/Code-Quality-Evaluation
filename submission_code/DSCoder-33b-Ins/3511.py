class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        total_coins = x + y
        if total_coins % 2 == 0:
            return "Alice"
        else:
            return "Bob"