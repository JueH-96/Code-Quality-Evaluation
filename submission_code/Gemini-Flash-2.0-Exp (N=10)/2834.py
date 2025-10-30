class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        positions = {}
        for num in nums:
            positions[num] = positions.get(num, 0) + 1
        
        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]
            
            if from_pos in positions:
                count = positions[from_pos]
                del positions[from_pos]
                positions[to_pos] = positions.get(to_pos, 0) + count
        
        return sorted(list(positions.keys()))