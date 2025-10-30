class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        from collections import defaultdict
        # Count how many marbles are in each position initially
        positions = defaultdict(int)
        for pos in nums:
            positions[pos] += 1
        
        # For each move, transfer the marbles from moveFrom[i] to moveTo[i]
        for f, t in zip(moveFrom, moveTo):
            count = positions[f]
            # Remove the marbles from the original position
            del positions[f]
            # Add them to the new position
            positions[t] += count
        
        # Return the sorted list of occupied positions
        return sorted(positions.keys())