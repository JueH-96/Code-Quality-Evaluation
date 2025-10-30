from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        counts = {}
        ans = []
        for i in range(len(nums)):
            num = nums[i]
            f = freq[i]
            counts[num] = counts.get(num, 0) + f
            if counts[num] == 0:
                del counts[num]
            max_freq = 0
            if counts:
                max_freq = max(counts.values())
            ans.append(max_freq)
        return ans