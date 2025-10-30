class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        
        # If k >= n-1, the player with max skill will win
        # because they'll beat everyone once they're at the front
        if k >= n - 1:
            max_skill = max(skills)
            return skills.index(max_skill)
        
        # Create a queue with player indices
        from collections import deque
        queue = deque(range(n))
        
        # Track consecutive wins
        consecutive_wins = 0
        current_winner = queue[0]
        
        while True:
            # Get the two players at the front
            player1 = queue.popleft()
            player2 = queue.popleft()
            
            # Determine winner
            if skills[player1] > skills[player2]:
                winner = player1
                loser = player2
            else:
                winner = player2
                loser = player1
            
            # Update consecutive wins
            if winner == current_winner:
                consecutive_wins += 1
            else:
                current_winner = winner
                consecutive_wins = 1
            
            # Check if we have a winner
            if consecutive_wins == k:
                return current_winner
            
            # Winner stays at front, loser goes to back
            queue.appendleft(winner)
            queue.append(loser)