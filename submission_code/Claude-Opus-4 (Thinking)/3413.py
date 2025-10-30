class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        
        # If k is large enough, the player with max skill will eventually win
        if k >= n - 1:
            return skills.index(max(skills))
        
        # Simulate the competition process
        from collections import deque
        
        # Create a queue with player indices
        queue = deque(range(n))
        consecutive_wins = 0
        current_winner = None
        
        while consecutive_wins < k:
            # Get the first two players
            player1 = queue.popleft()
            player2 = queue.popleft()
            
            # Determine winner based on skill levels
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
            
            # Winner stays at front, loser goes to back
            queue.appendleft(winner)
            queue.append(loser)
        
        return current_winner