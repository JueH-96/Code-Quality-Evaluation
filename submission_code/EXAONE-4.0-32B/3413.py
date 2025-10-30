from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        if k >= n:
            return skills.index(max(skills))
        
        dq = deque(range(n))
        wins = 0
        
        while wins < k:
            a = dq.popleft()
            b = dq.popleft()
            if skills[a] > skills[b]:
                wins += 1
                dq.appendleft(a)
                dq.append(b)
            else:
                wins = 1
                dq.appendleft(b)
                dq.append(a)
                
        return dq[0]