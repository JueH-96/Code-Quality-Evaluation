from typing import List
from collections import deque

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        max_skill_val = max(skills)
        queue = deque(range(len(skills)))
        # First game
        a = queue.popleft()
        b = queue.popleft()
        if skills[a] > skills[b]:
            W = a
            L = b
        else:
            W = b
            L = a
        current_winner = W
        current_streak = 1
        if current_streak >= k:
            return W
        if skills[W] == max_skill_val:
            return W
        queue.appendleft(W)
        queue.append(L)
        
        # Subsequent games
        while True:
            a = queue.popleft()  # Current winner
            b = queue.popleft()  # Challenger
            if skills[a] > skills[b]:
                W = a
                L = b
                # Same winner
                current_streak += 1
            else:
                W = b
                L = a
                # New winner
                current_streak = 1
            if current_streak >= k:
                return W
            if skills[W] == max_skill_val:
                return W
            queue.appendleft(W)
            queue.append(L)