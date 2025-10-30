class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Track the occupied positions
        occupied_positions = set(nums)
        
        # Perform the moves
        for i in range(len(moveFrom)):
            occupied_positions.remove(moveFrom[i])
            occupied_positions.add(moveTo[i])
        
        # Return the sorted list of occupied positions
        return sorted(occupied_positions)