class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        stones = n - 10
        if stones < 0:
            return False
        current_move = 9
        alice_turn = False  # Bob's turn after Alice's first move
        while True:
            if stones < current_move:
                return not alice_turn
            stones -= current_move
            current_move -= 1
            alice_turn = not alice_turn
            if current_move < 0:
                break