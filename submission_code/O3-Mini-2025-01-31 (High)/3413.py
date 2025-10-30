from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        # If k is huge, then eventually the player with the highest skill wins.
        # Since the player with maximum skill will never lose,
        # if k >= n - 1 then this player would have defeated all the others.
        if k >= n - 1:
            max_skill = max(skills)
            return skills.index(max_skill)
        
        # Otherwise, simulate the competition.
        candidate_skill = skills[0]
        candidate_index = 0
        win_count = 0
        
        # Process the players in order (simulate one cycle)
        for i in range(1, n):
            if candidate_skill > skills[i]:
                win_count += 1
            else:
                candidate_skill = skills[i]
                candidate_index = i
                win_count = 1  # New candidate wins the match against the former candidate.
            
            if win_count == k:
                return candidate_index
        
        # If the loop finishes without a candidate accumulating k wins,
        # then the candidate is the best (maximum skilled) player.
        return candidate_index

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    print(sol.findWinningPlayer([4,2,6,3,9], 2))  # Expected output: 2
    print(sol.findWinningPlayer([2,5,4], 3))        # Expected output: 1