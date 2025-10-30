class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        if k >= n:
            return skills.index(max(skills))
        
        queue = list(range(n))
        current_winner = queue[0]
        consecutive_wins = 0
        
        while True:
            next_player = queue[1]
            if skills[current_winner] > skills[next_player]:
                consecutive_wins += 1
                queue.pop(1)
                queue.append(next_player)
            else:
                consecutive_wins = 1
                queue.pop(0)
                queue.append(current_winner)
                current_winner = next_player
            if consecutive_wins == k:
                return current_winner