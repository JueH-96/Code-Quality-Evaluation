from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        """
        Finds the winner of an array game by efficiently simulating the process.
        """
        n = len(skills)
        
        current_winner_idx = 0
        win_streak = 0

        # We simulate the games. The player at `current_winner_idx` is at the front.
        # The player at index `i` is the next challenger from the initial lineup.
        # This loop will determine the winner in at most n-1 games.
        for i in range(1, n):
            # If the current winner is stronger than the challenger, they win again.
            if skills[current_winner_idx] > skills[i]:
                win_streak += 1
            # Otherwise, the challenger becomes the new winner.
            else:
                current_winner_idx = i
                win_streak = 1
            
            # If a player has won k games in a row, they are the competition winner.
            if win_streak == k:
                return current_winner_idx

        # If the loop completes, no one won k games in the first pass.
        # The player who is the `current_winner_idx` at this point must be the
        # strongest player overall, as they have survived all initial challenges.
        # They will never lose again and are guaranteed to eventually win k games.
        return current_winner_idx