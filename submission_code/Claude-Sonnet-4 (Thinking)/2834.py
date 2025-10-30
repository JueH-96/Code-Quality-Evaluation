class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Track all occupied positions using a set
        occupied = set(nums)
        
        # Process each move
        for from_pos, to_pos in zip(moveFrom, moveTo):
            occupied.remove(from_pos)  # All marbles leave this position
            occupied.add(to_pos)       # All marbles arrive at this position
        
        # Return sorted list of occupied positions
        return sorted(list(occupied))