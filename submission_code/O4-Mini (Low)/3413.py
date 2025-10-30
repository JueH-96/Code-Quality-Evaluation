from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        # Pair each player index with its skill
        dq = deque((i, s) for i, s in enumerate(skills))
        # Track the current champion and its consecutive wins
        # We'll start with no champion and 0 wins
        current_idx, current_skill = dq.popleft()
        wins = 0
        # Precompute the highest skill: if someone reaches this,
        # they will never be beaten afterward.
        max_skill = max(skills)
        
        # Simulate until someone reaches k wins in a row
        while True:
            # If current is already the overall best and k is large,
            # it will just keep winning
            if current_skill == max_skill and wins + 1 >= k:
                return current_idx
            
            # Next challenger
            challenger_idx, challenger_skill = dq.popleft()
            
            # Determine the outcome
            if current_skill > challenger_skill:
                # current wins again
                wins += 1
                # loser goes to the back
                dq.append((challenger_idx, challenger_skill))
            else:
                # challenger becomes the new current champion
                wins = 1
                # loser (old current) goes to the back
                dq.append((current_idx, current_skill))
                current_idx, current_skill = challenger_idx, challenger_skill
            
            # Check if we've reached k wins
            if wins >= k:
                return current_idx