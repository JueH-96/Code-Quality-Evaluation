from collections import Counter

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        marble_positions = Counter(nums)

        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]

            if from_pos in marble_positions:
                count = marble_positions[from_pos]
                if count > 0:
                    marble_positions[from_pos] -= count
                    marble_positions[to_pos] += count

        occupied_positions = sorted([pos for pos, count in marble_positions.items() if count > 0])
        return occupied_positions