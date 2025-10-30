from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = nums.length
        total_sum = 0

        for i in range(n):
            distinct_count = 0
            count = [0] * 101  # Since nums[i] <= 100
            for j in range(i, n):
                if count[nums[j]] == 0:
                    distinct_count += 1
                count[nums[j]] += 1
                total_sum += distinct_count * distinct_count

        return total_sum