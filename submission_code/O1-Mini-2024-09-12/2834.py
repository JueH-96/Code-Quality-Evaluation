from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        positions = {}
        for num in nums:
            positions[num] = positions.get(num, 0) + 1

        for f, t in zip(moveFrom, moveTo):
            count = positions.get(f, 0)
            if count > 0:
                positions[f] -= count
                if positions[f] == 0:
                    del positions[f]
                positions[t] = positions.get(t, 0) + count

        return sorted(positions.keys())