class Solution:
    def canAliceWin(self, n: int) -> bool:
        remaining = n
        current_move = 10
        alice_turn = True
        
        while True:
            if remaining < current_move:
                return not alice_turn
            remaining -= current_move
            if remaining == 0:
                return alice_turn
            current_move -= 1
            alice_turn = not alice_turn