class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        from collections import deque
        
        n = len(skills)
        queue = deque(range(n))
        
        current_winner = queue[0]
        consecutive_wins = 0
        max_skill = max(skills)
        
        while consecutive_wins < k:
            player1 = queue.popleft()
            player2 = queue.popleft()
            
            if skills[player1] > skills[player2]:
                winner = player1
                loser = player2
            else:
                winner = player2
                loser = player1
            
            # Winner goes to front, loser goes to back
            queue.appendleft(winner)
            queue.append(loser)
            
            if winner == current_winner:
                consecutive_wins += 1
            else:
                current_winner = winner
                consecutive_wins = 1
            
            # Optimization: if current winner has max skill, they'll win all future games
            if skills[current_winner] == max_skill:
                break
        
        return current_winner