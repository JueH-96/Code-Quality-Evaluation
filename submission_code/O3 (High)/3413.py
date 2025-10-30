from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        """
        Simulate the competition just until every player has appeared once
        in the first position of the queue. During this first pass the first
        player who reaches k consecutive victories is returned.
        
        If nobody reaches k victories in that time, the current player at the
        head of the queue must be the one whose skill is the maximum among all
        players. Since this player can no longer be beaten, he will eventually
        accumulate any remaining wins that are still needed, so he is the
        overall winner.
        
        Time complexity  : O(n)
        Auxiliary memory : O(1)
        """
        n = len(skills)
        
        # Player that is currently at the head of the queue
        current = 0
        # How many consecutive matches the current player has already won
        win_streak = 0
        
        for i in range(1, n):
            if skills[current] > skills[i]:
                # current beats the i-th player
                win_streak += 1
            else:
                # i-th player beats current and becomes the new leader
                current = i
                win_streak = 1    # the new leader has just won his first game
            
            if win_streak == k:
                return current
        
        # Nobody reached k wins during the first pass.
        # The player at `current` is the one with the maximum skill, therefore
        # he will eventually obtain k consecutive victories.
        return current