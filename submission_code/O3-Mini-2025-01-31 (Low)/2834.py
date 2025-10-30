class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        from collections import Counter
        
        # Count the initial marbles at each position
        position_count = Counter(nums)
        
        # Process each move operation
        for frm, to in zip(moveFrom, moveTo):
            count = position_count.get(frm, 0)
            if count > 0:
                position_count[to] = position_count.get(to, 0) + count
                # Remove the marbles from the source position
                del position_count[frm]
                
        # Return the sorted occupied positions
        return sorted(position_count.keys())