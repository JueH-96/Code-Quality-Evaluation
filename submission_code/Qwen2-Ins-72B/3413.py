from collections import deque

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        queue = deque(range(len(skills)))
        win_count = 0
        prev_winner = None
        
        while True:
            player1 = queue.popleft()
            player2 = queue.popleft()
            
            if skills[player1] > skills[player2]:
                winner = player1
                loser = player2
            else:
                winner = player2
                loser = player1
            
            if winner == prev_winner:
                win_count += 1
            else:
                win_count = 1
            
            if win_count == k:
                return winner
            
            prev_winner = winner
            queue.appendleft(winner)
            queue.append(loser)