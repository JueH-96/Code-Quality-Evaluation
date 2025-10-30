from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        queue = [i for i in range(len(skills))]
        wins = [0] * len(skills)
        while True:
            player1 = queue.pop(0)
            player2 = queue.pop(0)
            if skills[player1] > skills[player2]:
                wins[player1] += 1
                if wins[player1] == k:
                    return player1
                queue.append(player2)
                queue.append(player1)
            else:
                wins[player2] += 1
                if wins[player2] == k:
                    return player2
                queue.append(player1)
                queue.append(player2)