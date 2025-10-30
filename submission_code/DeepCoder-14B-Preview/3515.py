from collections import deque

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        if n == 0:
            return -1  # Should not happen as per constraints
        
        max_skill = max(skills)
        max_index = skills.index(max_skill)
        
        dq = deque(range(n))  # Initialize the deque with player indices
        
        current_streak = 0
        prev_winner = None
        
        while True:
            if len(dq) < 2:
                break  # This should not happen as per the problem constraints
            
            a = dq.popleft()
            b = dq.popleft()
            
            if skills[a] > skills[b]:
                winner = a
                loser = b
            else:
                winner = b
                loser = a
            
            dq.appendleft(winner)
            dq.append(loser)
            
            if prev_winner is None:
                current_streak = 1
                prev_winner = winner
            else:
                if winner == prev_winner:
                    current_streak += 1
                else:
                    current_streak = 1
                prev_winner = winner
            
            if current_streak >= k:
                return prev_winner
            
            # If the current winner is the max_index, they will dominate
            if prev_winner == max_index:
                return max_index