class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = list(range(n))
        consecutive_wins = 0
        
        # If k is larger than n, we only need to simulate until someone wins n consecutive games
        # because if someone can win n consecutive games, they must have the highest skill and
        # will eventually win k games in a row
        target_wins = min(k, n)
        
        while consecutive_wins < target_wins:
            player1_idx = queue[0]
            player2_idx = queue[1]
            
            if skills[player1_idx] > skills[player2_idx]:  # player1 wins
                winner_idx = player1_idx
                loser_idx = player2_idx
                consecutive_wins += 1  # Player 1 stays at the front, increment consecutive wins
            else:  # player2 wins
                winner_idx = player2_idx
                loser_idx = player1_idx
                consecutive_wins = 1  # New player at the front, reset consecutive wins to 1
            
            # Update the queue: winner at front, loser at end
            queue = [winner_idx] + queue[2:] + [loser_idx]
        
        return queue[0]  # Return the original index of the winner