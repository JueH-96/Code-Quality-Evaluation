class Solution:
    def canAliceWin(self, n: int) -> bool:
        stones = n
        turn = 0  # 0 for Alice, 1 for Bob
        stones_to_remove = 10
        
        while stones_to_remove > 0:
            if stones < stones_to_remove:
                # Current player can't make a move
                return turn == 1  # Alice wins if Bob can't move
            
            stones -= stones_to_remove
            stones_to_remove -= 1
            turn = 1 - turn
        
        # If we reach here, stones_to_remove is 0, meaning the last move was removing 1 stone
        # The next player would need to remove 0 stones, which is invalid, so they lose
        return turn == 1  # Alice wins if it's Bob's turn to make an invalid move