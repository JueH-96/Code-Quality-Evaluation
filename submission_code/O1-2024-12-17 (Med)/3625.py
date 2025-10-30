class Solution:
    def canAliceWin(self, n: int) -> bool:
        # The required stones to remove on each turn, starting with 10 for Alice,
        # then 9 for Bob, etc., down to 1.
        removals = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        
        # Current player: 0 for Alice, 1 for Bob
        current_player = 0
        stones = n
        
        # Simulate each move in order
        for r in removals:
            if stones < r:
                # Current player cannot make the required move, so they lose
                return (current_player == 1)  # If Bob can't move, Alice wins (True); if Alice can't move, False
            # Otherwise, remove the stones and switch player
            stones -= r
            current_player = 1 - current_player
        
        # If all 10 removals are possible, Bob takes the last stone
        # and the next move (removal of 0) is impossible for Alice, so Alice loses
        return False