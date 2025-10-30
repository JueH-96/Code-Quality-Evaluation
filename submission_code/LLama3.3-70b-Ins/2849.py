from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        def calculate_imbalance(subarray):
            sorted_subarray = sorted(subarray)
            imbalance = 0
            for i in range(len(sorted_subarray) - 1):
                if sorted_subarray[i + 1] - sorted_subarray[i] > 1:
                    imbalance += 1
            return imbalance

        total_imbalance = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums) + 1):
                subarray = nums[i:j]
                total_imbalance += calculate_imbalance(subarray)

        return total_imbalance