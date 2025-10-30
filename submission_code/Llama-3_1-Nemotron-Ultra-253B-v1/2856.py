from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        total_distinct = len(set(nums))
        n = len(nums)
        res = 0
        for i in range(n):
            freq = {}
            for j in range(i, n):
                num = nums[j]
                if num in freq:
                    freq[num] += 1
                else:
                    freq[num] = 1
                if len(freq) == total_distinct:
                    res += n - j
                    break
        return res