import collections
from typing import List

class Solution:
  def findWinningPlayer(self, skills: List[int], k: int) -> int:
    n = len(skills)

    # Optimization for large k:
    # If k is large enough, the player with the maximum skill will eventually win.
    # A player who wins n-1 games in a row must have defeated all other n-1 players.
    # This means they are the strongest player. Once they are at the front and have won n-1 games,
    # they will continue to win against all other players indefinitely.
    # Thus, if k >= n-1, the player with the maximum skill is the winner.
    # Constraints: n >= 2, so n-1 >= 1. k >= 1.
    if k >= n - 1:
        max_skill_val = -1
        winner_idx = -1
        # Find the player with the maximum skill.
        # Since all skill levels are unique, there's only one such player.
        for i in range(n):
            if skills[i] > max_skill_val:
                max_skill_val = skills[i]
                winner_idx = i
        return winner_idx

    # If k < n-1, simulation is needed.
    # A non-strongest player might win k games before the strongest player dominates.
    
    # Initialize a deque with (skill_value, original_index) for each player.
    player_queue = collections.deque()
    for i in range(n):
        player_queue.append((skills[i], i))

    # The first player in the queue is the initial 'current champion'.
    # (skill_value, original_index)
    current_champion_tuple = player_queue.popleft() 
    consecutive_wins = 0

    # Simulate games. Each iteration of the while loop is one game.
    # The loop terminates because either a player achieves k wins.
    # Since k < n-1, the number of games simulated is bounded by O(N).
    while True: 
        # The next player in queue is the opponent.
        opponent_tuple = player_queue.popleft()

        current_champion_skill = current_champion_tuple[0]
        opponent_skill = opponent_tuple[0]

        if current_champion_skill > opponent_skill:
            # Current champion wins.
            consecutive_wins += 1
            # Loser (opponent) goes to the end of the queue.
            player_queue.append(opponent_tuple) 
        else:
            # Opponent wins and becomes the new champion.
            # Old champion (now loser) goes to the end of the queue.
            player_queue.append(current_champion_tuple) 
            # Update current champion.
            current_champion_tuple = opponent_tuple
            # Reset consecutive wins for the new champion.
            consecutive_wins = 1 
        
        # Check if the current champion has won k games in a row.
        if consecutive_wins == k:
            return current_champion_tuple[1] # Return original_index