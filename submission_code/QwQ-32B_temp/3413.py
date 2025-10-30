from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        if n == 0:
            return -1
        max_skill = max(skills)
        max_player = skills.index(max_skill)
        q = deque(range(n))
        current_winner = None
        current_streak = 0
        
        while True:
            if q[0] == max_player:
                return max_player
            a = q.popleft()
            b = q.popleft()
            if skills[a] > skills[b]:
                winner = a
                loser = b
            else:
                winner = b
                loser = a
            # Update streak
            if winner == current_winner:
                current_streak += 1
            else:
                current_winner = winner
                current_streak = 1
            if current_streak >= k:
                return winner
            # Update the queue
            q.appendleft(winner)
            q.append(loser)