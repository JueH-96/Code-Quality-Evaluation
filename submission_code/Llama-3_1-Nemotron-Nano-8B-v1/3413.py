from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        max_skill = max(skills)
        max_index = skills.index(max_skill)
        q = deque(range(len(skills)))
        current = q.popleft()
        current_streak = 0
        
        while True:
            if current == max_index:
                return max_index
            challenger = q.popleft()
            if skills[current] > skills[challenger]:
                current_streak += 1
                if current_streak == k:
                    return current
                q.append(challenger)
            else:
                q.append(current)
                current = challenger
                current_streak = 1
                if current_streak == k:
                    return current