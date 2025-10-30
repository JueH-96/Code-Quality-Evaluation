class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        from collections import deque
        
        n = len(skills)
        # If k == 1, the very first match determines the winner (the first player to win just once).
        # But we can still handle it with the logic below (it will return immediately after the first win).
        
        # Keep track of the current champion (initially player 0) and how many consecutive wins they have.
        champion = 0
        consecutive_wins = 0
        
        # Prepare a queue with the rest of the players (1..n-1)
        q = deque(range(1, n))
        
        # We need at most n - 1 comparisons before the champion must be the player with the maximum skill.
        # Once we know the champion is the strongest player, if k hasn't been reached yet, that champion will
        # eventually get k consecutive wins (because they won't lose to anyone). So we can just return them.
        for _ in range(n - 1):
            # Next challenger is at the front of the queue
            challenger = q[0]
            
            if skills[champion] > skills[challenger]:
                # Champion wins again
                consecutive_wins += 1
                # Move the challenger (loser) to the back
                q.rotate(-1)
            else:
                # Challenger wins and becomes the new champion
                # Loser (old champion) goes to the back
                q[0] = champion  # Put old champion at front
                q.rotate(-1)     # Rotate so old champion goes to the back
                champion = challenger
                consecutive_wins = 1  # New champion has 1 consecutive win
            
            # Check if champion reached k consecutive wins
            if consecutive_wins == k:
                return champion
        
        # If we exit the loop, the current champion must be the strongest player overall.
        # They will eventually reach k consecutive wins, so return them now.
        return champion