from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        max_skill = max(skills)
        max_idx = skills.index(max_skill)
        n = len(skills)
        
        if k >= n - 1:
            return max_idx
        
        q = deque(range(n))
        current = q.popleft()
        consecutive = 0
        
        while True:
            if current == max_idx:
                return current
            if not q:
                break
            challenger = q.popleft()
            if skills[current] > skills[challenger]:
                consecutive += 1
                if consecutive == k:
                    return current
                q.append(challenger)
            else:
                q.append(current)
                current = challenger
                consecutive = 1
                if consecutive == k:
                    return current
        return max_idx