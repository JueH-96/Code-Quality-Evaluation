class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        # Initialize a set with the initial positions of marbles
        occupied = set(nums)
        
        # Perform each move from moveFrom[i] to moveTo[i]
        for frm, to in zip(moveFrom, moveTo):
            occupied.remove(frm)
            occupied.add(to)
        
        # Return the sorted list of occupied positions
        return sorted(occupied)