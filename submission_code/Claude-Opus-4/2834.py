class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Use a set to track occupied positions
        positions = set(nums)
        
        # Process each move
        for i in range(len(moveFrom)):
            # Remove all marbles from moveFrom[i]
            positions.discard(moveFrom[i])
            # Add position moveTo[i] (if not already present)
            positions.add(moveTo[i])
        
        # Return sorted list of occupied positions
        return sorted(list(positions))