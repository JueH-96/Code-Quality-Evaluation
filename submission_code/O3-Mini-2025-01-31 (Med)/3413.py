from collections import deque
from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        # If k is large (specifically, if k >= n-1),
        # the strongest (maximum skill) player will eventually win.
        max_skill = max(skills)
        max_index = skills.index(max_skill)
        if k >= n - 1:
            return max_index
        
        # Otherwise, simulate the games.
        # Represent each player as a tuple: (skill, original_index)
        dq = deque([(skills[i], i) for i in range(1, n)])
        cur_skill, cur_idx = skills[0], 0
        wins = 0
        
        # Simulate until a player wins k consecutive games.
        while wins < k:
            opp_skill, opp_idx = dq.popleft()
            if cur_skill > opp_skill:
                wins += 1
                # Loser goes to the end of the queue.
                dq.append((opp_skill, opp_idx))
            else:
                # New competitor wins the game, so reset streak.
                dq.append((cur_skill, cur_idx))
                cur_skill, cur_idx = opp_skill, opp_idx
                wins = 1
            # If the current winner has the maximum skill, they cannot lose.
            if cur_skill == max_skill:
                return cur_idx
        
        return cur_idx

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.findWinningPlayer([4, 2, 6, 3, 9], 2))  # Expected output: 2
    # Example 2:
    print(sol.findWinningPlayer([2, 5, 4], 3))        # Expected output: 1