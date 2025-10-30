from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        players = list(range(n))

        win_streak = [0] * n
        max_wins = 0

        while True:
            player1 = players[0]
            player2 = players[1]

            if skills[player1] > skills[player2]:
                winner = player1
                loser = player2
            else:
                winner = player2
                loser = player1

            win_streak[winner] += 1
            win_streak[loser] = 0

            if win_streak[winner] > max_wins:
                max_wins = win_streak[winner]

            if max_wins >= k:
                return winner

            players = [winner] + players[2:] + [loser]