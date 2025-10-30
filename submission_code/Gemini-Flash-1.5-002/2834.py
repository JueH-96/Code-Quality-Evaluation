class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        marble_positions = {}
        for num in nums:
            marble_positions[num] = marble_positions.get(num, 0) + 1

        for i in range(len(moveFrom)):
            if moveFrom[i] in marble_positions:
                count = marble_positions[moveFrom[i]]
                del marble_positions[moveFrom[i]]
                marble_positions[moveTo[i]] = marble_positions.get(moveTo[i], 0) + count

        occupied_positions = sorted(list(marble_positions.keys()))
        return occupied_positions