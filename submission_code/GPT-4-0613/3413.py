from typing import List
from collections import deque

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        queue = deque([(i, skills[i]) for i in range(len(skills))])
        winner, wins = queue.popleft()
        while wins < k:
            player, skill = queue.popleft()
            if skill > skills[winner]:
                queue.append((winner, 0))
                winner, wins = player, 1
            else:
                queue.append((player, 0))
                wins += 1
        return winner