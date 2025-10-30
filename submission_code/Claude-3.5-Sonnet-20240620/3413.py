class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = list(range(n))
        wins = 0
        current_winner = None

        while True:
            player1 = queue.pop(0)
            player2 = queue.pop(0)

            if skills[player1] > skills[player2]:
                winner, loser = player1, player2
            else:
                winner, loser = player2, player1

            queue.insert(0, winner)
            queue.append(loser)

            if winner == current_winner:
                wins += 1
                if wins == k:
                    return winner
            else:
                current_winner = winner
                wins = 1