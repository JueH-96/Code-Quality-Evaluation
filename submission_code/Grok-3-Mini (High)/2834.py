class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        occupied = set(nums)
        for from_pos, to_pos in zip(moveFrom, moveTo):
            occupied.remove(from_pos)
            occupied.add(to_pos)
        return sorted(list(occupied))