from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        max_skill = max(skills)
        max_index = skills.index(max_skill)
        n = len(skills)
        q = deque(range(n))  # Initial queue of player indices
        
        current_champion = None
        consecutive_wins = 0
        
        while True:
            # Check if current champion is the max and handle accordingly
            if current_champion == max_index:
                if consecutive_wins >= k:
                    return max_index
                else:
                    return max_index  # Max can't lose, will eventually reach k
            
            # Check if current champion has already won k times
            if consecutive_wins >= k:
                return current_champion
            
            # Process the next battle
            if current_champion is None:
                # First battle: take the first two players
                a = q.popleft()
                b = q.popleft()
                if skills[a] > skills[b]:
                    winner, loser = a, b
                else:
                    winner, loser = b, a
                current_champion = winner
                consecutive_wins = 1
                q.appendleft(winner)
                q.append(loser)
            else:
                # Current champion is at front, next challenger is the second element
                challenger = q[1]
                if skills[current_champion] > skills[challenger]:
                    # Champion wins, update consecutive wins and rearrange deque
                    consecutive_wins += 1
                    # Remove champion and challenger, then append them in new order
                    q.popleft()
                    q.popleft()
                    q.appendleft(current_champion)
                    q.append(challenger)
                else:
                    # Challenger wins, update champion and reset consecutive wins
                    q.popleft()
                    q.popleft()
                    q.appendleft(challenger)
                    q.append(current_champion)
                    current_champion = challenger
                    consecutive_wins = 1