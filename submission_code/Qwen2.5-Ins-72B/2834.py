from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        positions = set(nums)
        
        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]
            
            if from_pos in positions:
                positions.remove(from_pos)
                positions.add(to_pos)
        
        return sorted(positions)