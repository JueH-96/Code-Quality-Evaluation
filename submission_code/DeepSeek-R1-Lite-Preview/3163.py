from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        for i in range(n):
            freq = {}
            distinct_count = 0
            for j in range(i, n):
                if nums[j] not in freq:
                    distinct_count += 1
                    freq[nums[j]] = 1
                else:
                    freq[nums[j]] += 1
                total_sum += distinct_count ** 2
        return total_sum