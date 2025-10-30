class Solution:
    def canAliceWin(self, n: int) -> bool:
        current = n
        required = 10
        alice_turn = True
        while required <= current:
            current -= required
            required -= 1
            alice_turn = not alice_turn
        return not alice_turn