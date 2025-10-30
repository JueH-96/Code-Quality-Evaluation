class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = list(range(n))
        win_count = 0
        current_winner = -1

        if k >= n - 1:
            max_skill_index = 0
            max_skill = -1
            for i in range(n):
                if skills[i] > max_skill:
                    max_skill = skills[i]
                    max_skill_index = i
            return max_skill_index

        while win_count < k:
            player1_index = queue.pop(0)
            player2_index = queue.pop(0)

            player1_skill = skills[player1_index]
            player2_skill = skills[player2_index]

            if player1_skill > player2_skill:
                winner_index = player1_index
                loser_index = player2_index
            else:
                winner_index = player2_index
                loser_index = player1_index

            queue.insert(0, winner_index)
            queue.append(loser_index)

            if winner_index == current_winner:
                win_count += 1
            else:
                current_winner = winner_index
                win_count = 1
            
            if win_count == k:
                return current_winner