from typing import List
from collections import deque

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        # If k is large enough, the strongest player will inevitably win
        if k >= n - 1:
            # Return the index of the maximum skill
            max_skill = max(skills)
            return skills.index(max_skill)
        
        # Initialize the champion as the first player
        champion_skill = skills[0]
        champion_idx = 0
        # Build a queue of challengers: (skill, original_index)
        queue = deque((skills[i], i) for i in range(1, n))
        
        consecutive_wins = 0
        
        # Simulate until someone wins k times in a row
        while True:
            challenger_skill, challenger_idx = queue[0]
            
            if champion_skill > challenger_skill:
                # Champion wins this round
                consecutive_wins += 1
                # Loser goes to back of the queue
                queue.popleft()
                queue.append((challenger_skill, challenger_idx))
            else:
                # Challenger becomes the new champion
                consecutive_wins = 1
                queue.popleft()
                queue.append((champion_skill, champion_idx))
                champion_skill, champion_idx = challenger_skill, challenger_idx
            
            # Check if champion has achieved k consecutive wins
            if consecutive_wins == k:
                return champion_idx