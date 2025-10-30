class Solution:
    def canAliceWin(self, n: int) -> bool:
        stones = n
        last_player = None
        player = 'Alice'
        next_move = 10

        while True:
            if stones >= next_move:
                stones -= next_move
                last_player = player
                next_move -= 1
                if next_move <= 0:
                    break
                # Switch player
                player = 'Bob' if player == 'Alice' else 'Alice'
            else:
                break

        return last_player == 'Alice'