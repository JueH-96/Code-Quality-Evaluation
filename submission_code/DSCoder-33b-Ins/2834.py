class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        marble_positions = set(nums)
        for i in range(len(moveFrom)):
            if moveFrom[i] in marble_positions:
                marble_positions.remove(moveFrom[i])
                marble_positions.add(moveTo[i])
        return sorted(list(marble_positions))