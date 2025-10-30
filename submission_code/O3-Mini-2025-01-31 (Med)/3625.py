class Solution:
    def canAliceWin(self, n: int) -> bool:
        # turn = 0 represents Alice's turn, turn = 1 represents Bob's turn
        turn = 0
        move = 10  # Alice starts by removing exactly 10 stones
        while True:
            if n < move:
                # If the current player cannot remove move stones, that player loses.
                # If turn is 0 (Alice) then she loses. Otherwise, Bob loses and Alice wins.
                return turn != 0
            n -= move
            move -= 1  # each subsequent turn, the removal amount is decreased by 1
            turn = 1 - turn  # toggle turns

# You can test the solution with the following code:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.canAliceWin(12))  # Expected output: True, because after Alice's move, Bob cannot move.
    # Example 2:
    print(sol.canAliceWin(1))   # Expected output: False, because Alice cannot remove 10 stones.