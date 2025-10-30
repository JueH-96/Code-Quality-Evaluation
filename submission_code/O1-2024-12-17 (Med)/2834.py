class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Count how many marbles are in each initial position
        positions = defaultdict(int)
        for pos in nums:
            positions[pos] += 1
        
        # For each move, shift the marbles from moveFrom[i] to moveTo[i]
        for f, t in zip(moveFrom, moveTo):
            count = positions[f]
            if count > 0:
                positions[f] -= count
                positions[t] += count
                # If no marbles remain at position f, remove it from dict
                if positions[f] == 0:
                    del positions[f]
        
        # Return the sorted list of occupied positions
        return sorted(positions.keys())