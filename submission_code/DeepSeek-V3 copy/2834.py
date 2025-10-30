class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Initialize a dictionary to keep track of the counts of marbles at each position
        position_counts = {}
        for num in nums:
            if num in position_counts:
                position_counts[num] += 1
            else:
                position_counts[num] = 1
        
        # Perform the moves
        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]
            if from_pos in position_counts:
                # Move all marbles from from_pos to to_pos
                count = position_counts[from_pos]
                if to_pos in position_counts:
                    position_counts[to_pos] += count
                else:
                    position_counts[to_pos] = count
                # Remove the from_pos from the dictionary
                del position_counts[from_pos]
        
        # Extract the occupied positions and sort them
        occupied_positions = sorted(position_counts.keys())
        return occupied_positions