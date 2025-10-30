class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        positions = set(nums)
        
        for src, dest in zip(moveFrom, moveTo):
            positions.remove(src)
            positions.add(dest)
        
        return sorted(positions)