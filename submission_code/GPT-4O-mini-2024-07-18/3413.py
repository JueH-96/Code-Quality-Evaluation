from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = deque(range(n))  # Initialize the queue with player indices
        win_count = [0] * n  # To count consecutive wins for each player
        
        while True:
            player1 = queue.popleft()  # First player in the queue
            player2 = queue.popleft()  # Second player in the queue
            
            if skills[player1] > skills[player2]:
                winner = player1
                loser = player2
            else:
                winner = player2
                loser = player1
            
            win_count[winner] += 1  # Increment the win count for the winner
            
            if win_count[winner] == k:  # Check if the winner has won k games in a row
                return winner
            
            # Reset the win count for the loser
            win_count[loser] = 0
            
            # Winner stays at the front, loser goes to the end
            queue.appendleft(winner)
            queue.append(loser)