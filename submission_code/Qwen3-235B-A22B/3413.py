from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        max_skill = -1
        max_idx = -1
        
        # Find the player with the maximum skill and their index
        for i in range(n):
            if skills[i] > max_skill:
                max_skill = skills[i]
                max_idx = i
        
        # If k >= n, the maximum skill player will always win eventually
        if k >= n:
            return max_idx
        
        current_champ = 0
        wins = 0
        
        # Iterate through each challenger in order
        for i in range(1, n):
            if skills[i] > skills[current_champ]:
                current_champ = i
                wins = 1
                if wins == k:
                    return current_champ
            else:
                wins += 1
                if wins == k:
                    return current_champ
        
        # If no one reached k wins, return the maximum skill player (champ after all processed)
        return current_champ