from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = list(range(n))
        win_count = [0] * n
        
        while True:
            player1 = queue.pop(0)
            player2 = queue.pop(0)
            
            if skills[player1] > skills[player2]:
                winner = player1
                loser = player2
            else:
                winner = player2
                loser = player1
            
            win_count[winner] += 1
            
            if win_count[winner] == k:
                return winner
            
            queue.insert(0, winner)
            queue.append(loser)