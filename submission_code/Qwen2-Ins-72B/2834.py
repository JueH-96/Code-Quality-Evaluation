class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        positions = set(nums)
        for mf, mt in zip(moveFrom, moveTo):
            positions.remove(mf)
            positions.add(mt)
        return sorted(positions)