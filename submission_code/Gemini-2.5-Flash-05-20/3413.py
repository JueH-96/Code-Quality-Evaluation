import collections
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)

        # Optimization for large k:
        # If k is greater than or equal to n-1, the player with the maximum skill
        # will eventually become the champion and defeat all other players.
        # Once they are the champion, they will never be defeated and will
        # accumulate k wins.
        # This handles cases where k is very large (e.g., 10^9) efficiently,
        # as a player cannot sustain a streak longer than n-1 against unique opponents
        # unless they are the absolute strongest.
        if k >= n - 1:
            max_skill = -1
            max_skill_player_idx = -1
            for i in range(n):
                if skills[i] > max_skill:
                    max_skill = skills[i]
                    max_skill_player_idx = i
            return max_skill_player_idx

        # Simulate the game using a deque for the queue of players.
        # The deque stores player *initial indices*.
        # Initial queue: [0, 1, 2, ..., n-1]
        q = collections.deque(range(n))

        # current_winner_idx stores the initial index of the player currently
        # holding the winning streak.
        # current_win_streak stores the number of consecutive wins for current_winner_idx.
        current_winner_idx = -1
        current_win_streak = 0

        # Start the first game
        # The first two players are player0 and player1 from the original queue.
        # After this initial setup, the winner will be identified, and subsequent
        # opponents will always be taken from the front of the deque.
        player1_idx_in_game = q.popleft()  # Player who was initially at index 0
        player2_idx_in_game = q.popleft()  # Player who was initially at index 1

        # The loop continues as long as no player has achieved k consecutive wins.
        # The total number of games simulated will be O(N) because the champion's
        # skill level strictly increases if they change, and the maximum skill
        # player will eventually dominate.
        while True:
            skill1 = skills[player1_idx_in_game]
            skill2 = skills[player2_idx_in_game]

            winner_idx = -1
            loser_idx = -1

            if skill1 > skill2:
                winner_idx = player1_idx_in_game
                loser_idx = player2_idx_in_game
            else: # skill2 > skill1 because all skills are unique
                winner_idx = player2_idx_in_game
                loser_idx = player1_idx_in_game
            
            # The loser goes to the end of the queue.
            q.append(loser_idx)

            # Update the win streak for the current champion.
            if winner_idx == current_winner_idx:
                current_win_streak += 1
            else:
                # A new player has become the champion, or it's the very first game
                # where current_winner_idx was initialized to -1.
                current_winner_idx = winner_idx
                current_win_streak = 1
            
            # Check if the current champion has won k games in a row.
            if current_win_streak == k:
                return current_winner_idx
            
            # Prepare for the next game:
            # The winner of the current game (current_winner_idx) stays at the front
            # (conceptually, they are player1_idx_in_game for the next round).
            # The next opponent is the player at the front of the actual queue (deque).
            player1_idx_in_game = current_winner_idx
            player2_idx_in_game = q.popleft()