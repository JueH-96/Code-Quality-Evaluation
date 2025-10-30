class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Use a set to track occupied positions
        occupied_positions = set(nums)
        
        # Process each move
        for from_pos, to_pos in zip(moveFrom, moveTo):
            # Remove marbles from the original position
            if from_pos in occupied_positions:
                occupied_positions.remove(from_pos)
            # Add marbles to the new position
            occupied_positions.add(to_pos)
        
        # Return the sorted list of occupied positions
        return sorted(occupied_positions)