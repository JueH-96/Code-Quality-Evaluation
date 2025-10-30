class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        
        # If k is larger than or equal to n-1, the player with the highest skill will eventually win
        if k >= n - 1:
            return skills.index(max(skills))
        
        # Initialize the queue with player indices
        queue = list(range(n))
        
        # Initialize the current winner and win streak
        current_winner = queue[0]
        win_streak = 0
        
        while win_streak < k:
            player1 = queue.pop(0)
            player2 = queue.pop(0)
            
            if skills[player1] > skills[player2]:
                # Player1 wins
                queue.insert(0, player1)
                queue.append(player2)
                if current_winner == player1:
                    win_streak += 1
                else:
                    current_winner = player1
                    win_streak = 1
            else:
                # Player2 wins
                queue.insert(0, player2)
                queue.append(player1)
                if current_winner == player2:
                    win_streak += 1
                else:
                    current_winner = player2
                    win_streak = 1
        
        return current_winner