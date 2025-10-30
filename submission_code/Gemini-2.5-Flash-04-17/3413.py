import collections
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)

        # Optimization for large k
        # If k is large enough (specifically, k >= n-1), the player with the maximum skill
        # will eventually reach the front and win k consecutive games.
        # The player with the maximum skill can defeat any opponent. Once they reach the front
        # (which they are guaranteed to do, as players with lower skill lose and move to the back),
        # they will stay at the front, accumulating wins indefinitely.
        # To defeat every other player, they need to play against the n-1 other players.
        # Once they have defeated all n-1 opponents, their consecutive win streak will be at least n-1.
        # If k is greater than or equal to n-1, this streak is long enough or will grow beyond n-1 to reach k.
        # Therefore, if k >= n-1, the winner is the player with the maximum skill.
        if k >= n - 1:
            max_skill = -1
            max_skill_index = -1
            for i in range(n):
                if skills[i] > max_skill:
                    max_skill = skills[i]
                    max_skill_index = i
            return max_skill_index

        # Simulate the competition using a deque when k is relatively small (k < n-1)
        # Store pairs of (skill, initial_index) to keep track of the original player index.
        q = collections.deque([(skills[i], i) for i in range(n)])

        # Initialize the current winner and their consecutive win count
        # The first player in the queue is the initial potential winner
        current_winner_skill, current_winner_index = q.popleft()
        consecutive_wins = 0

        # Simulate games until a winner is found
        # This loop is guaranteed to terminate because k is finite.
        # If the current winner is not the max skill player, they will eventually lose
        # to a stronger player, who then takes the front. This process brings
        # increasingly stronger players to the front until the max skill player arrives.
        # Once the max skill player is at the front (which happens eventually), they will
        # win every game and their streak will grow. Since we handle large k with optimization,
        # this loop runs only when k < n-1, in which case a winner is found within a number
        # of games that is expected to be acceptable for the given constraints.
        # The total number of games in the simulation phase seems bounded by a polynomial in n.
        while True:
            # Get the second player from the queue
            p2_skill, p2_index = q.popleft()

            # Determine the winner and loser based on skills
            if current_winner_skill > p2_skill:
                winner_skill, winner_index = current_winner_skill, current_winner_index
                loser_skill, loser_index = p2_skill, p2_index
            else: # skills are unique, so p2_skill must be greater than current_winner_skill
                winner_skill, winner_index = p2_skill, p2_index
                loser_skill, loser_index = current_winner_skill, current_winner_index

            # Update consecutive wins for the current winner
            if winner_index == current_winner_index:
                # The same player continues their winning streak
                consecutive_wins += 1
            else:
                # A new player broke the streak and became the winner
                current_winner_skill, current_winner_index = winner_skill, winner_index
                consecutive_wins = 1

            # Place players back into the queue according to the rules
            # The winner goes back to the front of the queue
            q.appendleft((winner_skill, winner_index))
            # The loser goes to the end of the queue
            q.append((loser_skill, loser_index))

            # Check if the winning condition is met
            if consecutive_wins == k:
                return current_winner_index