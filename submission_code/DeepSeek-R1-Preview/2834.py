class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        current_positions = set(nums)
        for frm, to in zip(moveFrom, moveTo):
            if frm == to:
                continue
            current_positions.remove(frm)
            current_positions.add(to)
        return sorted(current_positions)