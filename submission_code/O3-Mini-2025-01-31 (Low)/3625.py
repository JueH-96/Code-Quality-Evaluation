class Solution:
    def canAliceWin(self, n: int) -> bool:
        # Initial removal for Alice is 10 stones.
        current_removal = 10
        # Flag to indicate whose turn, True for Alice, False for Bob.
        isAliceTurn = True
        
        while True:
            # If there are not enough stones for the current turn, the player loses.
            if n < current_removal:
                return not isAliceTurn  # because the current player loses, so the opponent wins.
            # Deduct the stones removed.
            n -= current_removal
            # Decrease the removal count for the next turn.
            current_removal -= 1
            # Switch turn.
            isAliceTurn = not isAliceTurn