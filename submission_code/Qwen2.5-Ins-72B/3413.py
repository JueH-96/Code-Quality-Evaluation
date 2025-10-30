from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = list(range(n))
        win_count = 0
        current_winner = queue[0]
        
        while win_count < k:
            player1 = queue.pop(0)
            player2 = queue.pop(0)
            
            if skills[player1] > skills[player2]:
                winner = player1
                loser = player2
            else:
                winner = player2
                loser = player1
            
            if winner == current_winner:
                win_count += 1
            else:
                current_winner = winner
                win_count = 1
            
            queue.insert(0, winner)
            queue.append(loser)
            
            if win_count == k:
                return current_winner
            
            if win_count > n - 1:
                return current_winner
        
        return current_winner