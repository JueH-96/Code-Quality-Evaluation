class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Use a set to track occupied positions
        occupied = set(nums)
        
        # Process each move
        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]
            
            # Remove the source position (all marbles move away)
            occupied.discard(from_pos)
            
            # Add the destination position
            occupied.add(to_pos)
        
        # Return sorted list of occupied positions
        return sorted(occupied)