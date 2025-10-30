class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        occupied = set(nums)
        for frm, to in zip(moveFrom, moveTo):
            if frm == to:
                continue
            if frm in occupied:
                occupied.remove(frm)
                occupied.add(to)
        return sorted(occupied)