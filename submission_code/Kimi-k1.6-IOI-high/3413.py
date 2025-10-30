from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        max_skill = max(skills)
        max_idx = skills.index(max_skill)
        if k >= n - 1:
            return max_idx
        dq = deque(range(n))
        current_winner = None
        current_streak = 0
        while True:
            # Pop the first two players from the deque
            p1 = dq.popleft()
            p2 = dq.popleft()
            # Determine the winner
            if skills[p1] > skills[p2]:
                winner = p1
                loser = p2
            else:
                winner = p2
                loser = p1
            # Update current streak
            if winner == current_winner:
                current_streak += 1
            else:
                current_winner = winner
                current_streak = 1
            # Check if we have a winner with k consecutive wins
            if current_streak >= k:
                return winner
            # Update the deque with the winner at front and loser at end
            dq.appendleft(winner)
            dq.append(loser)