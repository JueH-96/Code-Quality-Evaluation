class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        occupied_positions = set(nums)
        
        for from_pos, to_pos in zip(moveFrom, moveTo):
            if from_pos in occupied_positions:
                occupied_positions.discard(from_pos)
                occupied_positions.add(to_pos)
        
        return sorted(occupied_positions)