class Solution:
    def canAliceWin(self, n: int) -> bool:
        removal = 10
        alice_turn = True
        
        while True:
            # If the required removal is not positive or there aren't enough stones to remove,
            # the current player cannot make a valid move and therefore loses.
            if removal <= 0 or n < removal:
                return not alice_turn
            
            # Remove the stones and switch turns
            n -= removal
            removal -= 1
            alice_turn = not alice_turn