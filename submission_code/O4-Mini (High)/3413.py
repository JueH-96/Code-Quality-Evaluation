from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        # current champion's skill and original index
        champ_skill = skills[0]
        champ_idx = 0
        # consecutive wins of the current champion
        consecutive_wins = 0
        
        # simulate by scanning the array once
        for i in range(1, n):
            if champ_skill > skills[i]:
                # champion wins this round
                consecutive_wins += 1
            else:
                # challenger becomes the new champion
                champ_skill = skills[i]
                champ_idx = i
                consecutive_wins = 1
            # if the champion has won k in a row, return their index
            if consecutive_wins == k:
                return champ_idx
        
        # if no one reached k in this pass, the champion is the max skill
        return champ_idx