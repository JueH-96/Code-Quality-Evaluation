from typing import List
from collections import deque

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        # Create a queue of players with their initial positions
        queue = deque([(i, skills[i]) for i in range(len(skills))])
        
        # Initialize the winner and their consecutive wins
        winner = None
        consecutive_wins = 0
        
        # Continue the competition until a player wins k games in a row
        while consecutive_wins < k:
            # Get the first two players in the queue
            player1 = queue.popleft()
            player2 = queue.popleft()
            
            # Determine the winner of the game
            if player1[1] > player2[1]:
                winner = player1[0]
                # Add the loser to the end of the queue
                queue.append(player2)
                # Add the winner back to the beginning of the queue
                queue.appendleft(player1)
                # Increment the consecutive wins
                consecutive_wins += 1
            else:
                winner = player2[0]
                # Add the loser to the end of the queue
                queue.append(player1)
                # Add the winner back to the beginning of the queue
                queue.appendleft(player2)
                # Reset the consecutive wins
                consecutive_wins = 1
        
        # Return the initial index of the winning player
        return winner