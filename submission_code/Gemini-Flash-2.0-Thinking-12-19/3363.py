from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        counts = {}
        ans = []
        for i in range(len(nums)):
            id_val = nums[i]
            freq_val = freq[i]
            counts[id_val] = counts.get(id_val, 0) + freq_val
            ids_to_remove = []
            for id_key, count in counts.items():
                if count <= 0:
                    ids_to_remove.append(id_key)
            for id_remove in ids_to_remove:
                del counts[id_remove]

            max_freq = 0
            if counts:
                max_freq = max(counts.values())
            ans.append(max_freq)
        return ans