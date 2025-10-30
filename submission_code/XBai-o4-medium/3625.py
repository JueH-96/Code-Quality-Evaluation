class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        m = n - 10
        current_k = 9
        is_alice_turn = False
        while True:
            if m < current_k:
                return not is_alice_turn
            else:
                m -= current_k
                current_k -= 1
                is_alice_turn = not is_alice_turn