class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        from collections import deque

        n = len(skills)
        
        # If k is 1, the very first match decides the winner, so we can handle that quickly.
        # But the general loop below also correctly handles k=1, so no special case needed.
        
        # Build an initial queue of player indices from 0..n-1
        players = deque(range(n))
        
        # Start with player 0 as the "current winner"
        winner_index = players.popleft()
        consecutive_wins = 0
        
        # The largest skill value's index:
        max_skill_index = max(range(n), key=lambda i: skills[i])
        
        # We will simulate matches until we have k consecutive wins,
        # or until the current winner is the max-skill player who cannot be beaten
        while True:
            # Get the next challenger
            challenger_index = players.popleft()
            
            # Determine the winner of the current game
            if skills[winner_index] > skills[challenger_index]:
                # Current winner wins again
                consecutive_wins += 1
                # Loser goes to the back
                players.append(challenger_index)
            else:
                # Challenger becomes the new winner
                players.append(winner_index)
                winner_index = challenger_index
                consecutive_wins = 1
            
            # Check if current winner has reached k consecutive wins
            if consecutive_wins == k:
                return winner_index
            
            # If the current winner has the maximum skill, it will never lose.
            # We can short-circuit by returning as soon as it wins k in a row.
            # But if k is large, we can skip directly once we know it's unstoppable:
            if winner_index == max_skill_index:
                return winner_index