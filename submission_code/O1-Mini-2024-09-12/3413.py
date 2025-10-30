from typing import List
from collections import deque

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        if k >= len(skills) -1:
            return skills.index(max(skills))
        
        dq = deque(range(len(skills)))
        current_winner = dq.popleft()
        win_count = 0
        
        while win_count < k:
            if not dq:
                return current_winner
            challenger = dq.popleft()
            if skills[current_winner] > skills[challenger]:
                win_count +=1
                dq.append(challenger)
            else:
                dq.append(current_winner)
                current_winner = challenger
                win_count =1
            # If the current winner is the maximum, it will win all future matches
            if skills[current_winner] == max(skills):
                return current_winner
        return current_winner