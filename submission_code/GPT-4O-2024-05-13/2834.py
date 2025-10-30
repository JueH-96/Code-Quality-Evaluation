class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Use a set to keep track of occupied positions
        occupied_positions = set(nums)
        
        # Process each move
        for frm, to in zip(moveFrom, moveTo):
            if frm in occupied_positions:
                occupied_positions.remove(frm)
                occupied_positions.add(to)
        
        # Return the sorted list of occupied positions
        return sorted(occupied_positions)