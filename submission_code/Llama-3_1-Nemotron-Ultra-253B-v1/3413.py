from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        max_idx = skills.index(max(skills))
        q = deque(range(len(skills)))
        current_winner = q.popleft()
        consecutive_wins = 0
        
        while True:
            next_player = q.popleft()
            if skills[current_winner] > skills[next_player]:
                consecutive_wins += 1
            else:
                current_winner, next_player = next_player, current_winner
                consecutive_wins = 1
            
            if current_winner == max_idx:
                return max_idx
            if consecutive_wins >= k:
                return current_winner
            
            q.appendleft(current_winner)
            q.append(next_player)