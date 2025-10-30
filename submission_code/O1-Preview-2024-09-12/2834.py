class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        positions = set(nums)
        for f, t in zip(moveFrom, moveTo):
            positions.remove(f)
            positions.add(t)
        return sorted(positions)