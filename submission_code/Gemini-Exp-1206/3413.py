from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = deque(range(n))
        wins = {}
        while True:
            player1 = queue.popleft()
            player2 = queue.popleft()
            if skills[player1] > skills[player2]:
                winner = player1
                loser = player2
            else:
                winner = player2
                loser = player1
            wins[winner] = wins.get(winner, 0) + 1
            if wins[winner] == k:
                return winner
            queue.appendleft(winner)
            queue.append(loser)