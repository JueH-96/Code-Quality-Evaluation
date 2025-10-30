from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        occupied_positions = set(nums)
        for start, end in zip(moveFrom, moveTo):
            occupied_positions.remove(start)
            occupied_positions.add(end)
        return sorted(list(occupied_positions))