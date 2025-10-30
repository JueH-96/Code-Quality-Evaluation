class Solution:
    def canAliceWin(self, n: int) -> bool:
        # True => Alice's turn, False => Bob's turn
        alice_turn = True
        # Removal sequence: 10, 9, 8, ..., 1
        for k in range(10, 0, -1):
            # If current player cannot remove k stones, they lose
            if n < k:
                # If it's Alice's turn, Alice loses (return False).
                # If it's Bob's turn, Bob loses => Alice wins (return True).
                return not alice_turn
            # Perform the removal
            n -= k
            # Switch turn
            alice_turn = not alice_turn
        # Given n <= 50 and sum(10..1) = 55 > n, we'll always return inside the loop.
        # But add a default return for completeness.
        return False