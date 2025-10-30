from typing import List
from collections import deque

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = deque(range(n))
        consecutive_wins = [0] * n  # Track consecutive wins for each player
        
        while True:
            # Get the first two players
            p1 = queue[0]
            p2 = queue[1]
            
            # Determine the winner and loser
            if skills[p1] > skills[p2]:
                winner = p1
                loser = p2
            else:
                winner = p2
                loser = p1
            
            # Update the queue: winner stays front, loser goes to the end
            queue.append(queue.popleft())  # Remove p1 and add to end
            if winner != queue[0]:
                queue.append(queue.popleft())  # Remove p2 and add to end
            else:
                queue.append(loser)
            
            # Update consecutive wins for the winner
            consecutive_wins[winner] += 1
            
            # Check if the winner has achieved k consecutive wins
            if consecutive_wins[winner] >= k:
                return winner