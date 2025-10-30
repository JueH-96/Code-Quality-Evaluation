from typing import List
from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count_map = defaultdict(int)
        freq_map = defaultdict(int)
        current_max = 0
        ans = []

        for num, f in zip(nums, freq):
            prev_count = count_map[num]
            new_count = prev_count + f
            count_map[num] = new_count

            # Update freq_map
            if prev_count > 0:
                freq_map[prev_count] -= 1
                if freq_map[prev_count] == 0:
                    del freq_map[prev_count]
            freq_map[new_count] += 1

            # Update current_max
            if new_count > current_max:
                current_max = new_count
            elif prev_count == current_max and freq_map.get(prev_count, 0) == 0:
                if freq_map:
                    current_max = max(freq_map.keys())
                else:
                    current_max = 0
            ans.append(current_max)

        return ans