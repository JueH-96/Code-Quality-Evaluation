class Solution:
    def canAliceWin(self, n: int) -> bool:
        # Alice starts by removing 10 stones
        stones_removed = 10
        current_turn_stones = 10

        # Simulate the game turns
        while n >= stones_removed:
            n -= current_turn_stones
            current_turn_stones -= 1
            if current_turn_stones == 0:
                current_turn_stones = 10

        # If it's Alice's turn to play and she can't make a move, she loses
        if current_turn_stones == 10:
            return False
        else:
            return True