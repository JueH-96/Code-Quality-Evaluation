class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        if k >= n - 1:
            return skills.index(max(skills))

        queue = list(range(n))
        wins = 0
        winner = -1

        while True:
            player1 = queue.pop(0)
            player2 = queue.pop(0)

            if skills[player1] > skills[player2]:
                if winner == player1:
                    wins += 1
                else:
                    winner = player1
                    wins = 1
                queue.insert(0, player1)
                queue.append(player2)
            else:
                if winner == player2:
                    wins += 1
                else:
                    winner = player2
                    wins = 1
                queue.insert(0, player2)
                queue.append(player1)

            if wins == k:
                return winner