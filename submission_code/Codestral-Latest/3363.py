from collections import defaultdict
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_count = defaultdict(int)
        max_freq = 0
        result = []

        for num, f in zip(nums, freq):
            if f > 0:
                id_count[num] += f
                if id_count[num] > max_freq:
                    max_freq = id_count[num]
            else:
                id_count[num] += f
                if id_count[num] == 0:
                    del id_count[num]
                if id_count[num] < max_freq:
                    max_freq = max(id_count.values()) if id_count else 0

            result.append(max_freq)

        return result