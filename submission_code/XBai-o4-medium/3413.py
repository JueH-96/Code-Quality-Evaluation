from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        current_max = skills[0]
        current_index = 0
        count = 0
        
        for i in range(1, n):
            if skills[i] > current_max:
                current_max = skills[i]
                current_index = i
                count = 1
            else:
                count += 1
            if count >= k:
                return current_index
        
        return current_index