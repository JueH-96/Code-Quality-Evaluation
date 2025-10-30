class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False

        stones = n
        stones_to_remove = 10

        while True:
            # Alice's turn
            if stones < stones_to_remove:
                return False  # Alice cannot move, Bob wins
            stones -= stones_to_remove
            stones_to_remove -= 1
            if stones_to_remove < 0:
                return True # Bob cannot make a move

            # Bob's turn
            if stones < stones_to_remove:
                return True   # Bob cannot move, Alice wins
            stones -= stones_to_remove
            stones_to_remove -= 1
            if stones_to_remove < 0:
                return False # Alice cannot make a move