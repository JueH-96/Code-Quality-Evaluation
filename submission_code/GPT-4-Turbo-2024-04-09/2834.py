class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Create a map to track the number of marbles at each position
        position_map = defaultdict(int)
        
        # Initialize the map with the initial positions of the marbles
        for pos in nums:
            position_map[pos] += 1
        
        # Process each move
        for from_pos, to_pos in zip(moveFrom, moveTo):
            if position_map[from_pos] > 0:
                # Move all marbles from from_pos to to_pos
                position_map[to_pos] += position_map[from_pos]
                position_map[from_pos] = 0
        
        # Collect all positions that have at least one marble
        occupied_positions = [pos for pos, count in position_map.items() if count > 0]
        
        # Return the sorted list of occupied positions
        return sorted(occupied_positions)