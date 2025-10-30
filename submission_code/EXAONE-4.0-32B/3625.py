class Solution:
    def canAliceWin(self, n: int) -> bool:
        stones = n
        move = 10
        turn = 0  # 0 for Alice, 1 for Bob
        while stones >= move and move > 0:
            stones -= move
            move -= 1
            turn = 1 - turn
        return turn == 1