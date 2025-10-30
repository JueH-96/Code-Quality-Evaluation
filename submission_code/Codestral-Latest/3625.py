class Solution:
    def canAliceWin(self, n: int) -> bool:
        # Alice starts by removing 10 stones on her first turn
        if n <= 10:
            return False

        # Calculate the number of stones left after Alice's first turn
        stones_left = n - 10

        # Alice and Bob will take turns removing stones
        # Alice will remove 10, 8, 6, 4, 2, 0 stones and so on
        # Bob will remove 9, 7, 5, 3, 1 stones and so on
        # The game ends when the number of stones left is less than or equal to 10

        # Alice wins if the number of stones left after her first turn is even
        return stones_left % 2 == 0