from typing import List
from collections import deque

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        players = deque(range(n))
        win_streak = 0
        winner = -1

        while win_streak < k:
            player1 = players.popleft()
            player2 = players.popleft()

            if skills[player1] > skills[player2]:
                players.appendleft(player1)
                players.append(player2)
                if player1 == winner:
                    win_streak += 1
                else:
                    winner = player1
                    win_streak = 1
            else:
                players.appendleft(player2)
                players.append(player1)
                if player2 == winner:
                    win_streak += 1
                else:
                    winner = player2
                    win_streak = 1

        return winner