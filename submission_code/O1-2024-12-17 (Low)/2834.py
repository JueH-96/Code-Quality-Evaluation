class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Count how many marbles are initially in each position
        position_counts = defaultdict(int)
        for position in nums:
            position_counts[position] += 1
        
        # Perform each move
        for i in range(len(moveFrom)):
            origin, target = moveFrom[i], moveTo[i]
            count = position_counts[origin]
            position_counts[origin] -= count
            # Remove the origin if no marbles remain there
            if position_counts[origin] == 0:
                del position_counts[origin]
            position_counts[target] += count
        
        # Return sorted list of positions that have at least one marble
        return sorted(position_counts.keys())