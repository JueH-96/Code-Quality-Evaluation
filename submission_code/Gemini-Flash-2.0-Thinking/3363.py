from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        n = len(nums)
        ans = []
        id_counts = {}

        for i in range(n):
            id_val = nums[i]
            freq_change = freq[i]

            id_counts[id_val] = id_counts.get(id_val, 0) + freq_change

            # Remove IDs with a count of 0
            keys_to_remove = [key for key, value in id_counts.items() if value == 0]
            for key in keys_to_remove:
                del id_counts[key]

            if not id_counts:
                ans.append(0)
            else:
                max_freq = 0
                for count in id_counts.values():
                    max_freq = max(max_freq, count)
                ans.append(max_freq)

        return ans