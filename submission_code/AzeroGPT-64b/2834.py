class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        seen = set(nums)
        for mf, mt in zip(moveFrom, moveTo):
            seen.remove(mf)
            seen.add(mt)
        return sorted(seen)