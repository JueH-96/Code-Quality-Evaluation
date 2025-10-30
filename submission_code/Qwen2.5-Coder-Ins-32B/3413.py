from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = list(range(n))
        wins = 0
        current_winner = queue.pop(0)
        
        while wins < k:
            next_player = queue.pop(0)
            if skills[current_winner] > skills[next_player]:
                wins += 1
                queue.append(next_player)
            else:
                wins = 1
                queue.append(current_winner)
                current_winner = next_player
            
            if wins == k:
                return current_winner
            
            if current_winner == max(queue):
                return current_winner
        
        return current_winner