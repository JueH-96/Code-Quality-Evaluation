from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = list(range(n))
        wins = [0] * n
        
        while True:
            first = queue.pop(0)
            second = queue.pop(0)
            
            if skills[first] > skills[second]:
                winner = first
                loser = second
            else:
                winner = second
                loser = first
            
            queue.insert(0, winner)
            queue.append(loser)
            wins[winner] += 1
            
            if wins[winner] == k:
                return winner