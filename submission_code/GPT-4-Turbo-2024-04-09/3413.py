class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        from collections import deque
        
        queue = deque(range(len(skills)))
        current_winner = queue.popleft()
        consecutive_wins = 0
        
        while True:
            next_player = queue.popleft()
            
            if skills[current_winner] > skills[next_player]:
                consecutive_wins += 1
                queue.append(next_player)
            else:
                queue.append(current_winner)
                current_winner = next_player
                consecutive_wins = 1
            
            if consecutive_wins == k:
                return current_winner