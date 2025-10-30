import math
from typing import List

class Solution:
    """
    Finds the initial index of the winning player in a competition.
    The winner is the first player to achieve k consecutive wins.
    The competition proceeds by having the first two players play,
    the winner stays at the front, and the loser goes to the end of the queue.
    """
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        """
        Args:
          skills: A list of unique integers representing player skill levels. 
                  The length of the list is n. Players are initially indexed 0 to n-1.
          k: The positive integer representing the number of consecutive wins 
             required to win the competition (k >= 1).

        Returns:
          The initial index (0 to n-1) of the winning player.
        """
        n = len(skills)

        # Edge case and Optimization: 
        # The maximum number of consecutive games any player can win is n-1 
        # (by defeating every other player). 
        # If k is greater than or equal to n-1, the player with the highest skill 
        # is guaranteed to eventually reach the front, win n-1 consecutive games, 
        # and thus satisfy the condition of winning k games.
        # In this scenario, we can simply find the player with the maximum skill.
        # This also correctly handles the case when k is extremely large (e.g., 10^9).
        if k >= n - 1:
            max_skill = -1
            max_index = -1
            # Iterate through the skills to find the maximum skill and its index
            for i in range(n):
                if skills[i] > max_skill:
                    max_skill = skills[i]
                    max_index = i
            return max_index

        # Simulation for the case when k < n - 1:
        # We can simulate the process efficiently without needing a full queue structure.
        # We only need to keep track of the player currently at the front (winner)
        # and their consecutive win count.
        
        current_winner_index = 0  # Start with player 0 at the front
        consecutive_wins = 0      # Initial win count is 0

        # Iterate through the other players (from index 1) as challengers.
        # This loop effectively simulates the games.
        for challenger_index in range(1, n):
            # Compare the skill of the current winner with the challenger.
            if skills[current_winner_index] > skills[challenger_index]:
                # The current winner wins the game. Increment their win streak.
                consecutive_wins += 1
            else:
                # The challenger wins. They become the new current winner.
                current_winner_index = challenger_index
                # The win streak resets to 1 for the new winner.
                consecutive_wins = 1 

            # Check if the current winner has achieved the required k wins.
            # Since k >= 1, this check is meaningful.
            if consecutive_wins == k:
                # If k wins are reached, this player is the winner. Return their index.
                return current_winner_index

        # If the loop completes without returning, it means no player achieved k 
        # consecutive wins during the first pass through all potential challengers (n-1 games).
        # This implies that the player who ended up at the front (current_winner_index) 
        # must be the player with the overall highest skill. Why? Because if any other 
        # player had higher skill, they would have eventually challenged and taken the lead.
        # A player with the highest skill, once at the front, will never lose again.
        # Since k < n-1, they might not have reached k wins *yet*, but they will continue 
        # winning against everyone who comes back from the end of the queue.
        # Therefore, this strongest player is the ultimate winner.
        return current_winner_index