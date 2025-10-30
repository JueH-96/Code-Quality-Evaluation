from collections import deque

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        if k >= n - 1:
            max_skill = -1
            winner_index = -1
            for i, skill in enumerate(skills):
                if skill > max_skill:
                    max_skill = skill
                    winner_index = i
            return winner_index

        queue = deque(range(n))
        wins = 0
        current_winner = -1

        while True:
            player1_index = queue.popleft()
            player2_index = queue.popleft()

            player1_skill = skills[player1_index]
            player2_skill = skills[player2_index]

            winner_index = player1_index if player1_skill > player2_skill else player2_index
            loser_index = player2_index if player1_skill > player2_skill else player1_index

            if winner_index == current_winner:
                wins += 1
            else:
                current_winner = winner_index
                wins = 1

            if wins == k:
                return current_winner

            queue.appendleft(winner_index)
            queue.append(loser_index)