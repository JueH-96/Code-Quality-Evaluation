from collections import defaultdict
from typing import List

class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        counter = defaultdict(int)
        for num in nums:
            counter[num] += 1
        
        for from_pos, to_pos in zip(moveFrom, moveTo):
            count = counter[from_pos]
            if count > 0:
                del counter[from_pos]
                counter[to_pos] += count
        
        return sorted(counter.keys())