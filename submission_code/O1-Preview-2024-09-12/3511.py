class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        player = 0  # 0 for Alice, 1 for Bob

        while x >= 1 and y >= 4:
            x -= 1
            y -= 4
            player = 1 - player  # Switch player

        if player == 0:
            return "Alice"
        else:
            return "Bob"