class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                sorted_subarray = sorted(subarray)
                imbalance_number = 0
                for k in range(len(sorted_subarray) - 1):
                    if sorted_subarray[k+1] - sorted_subarray[k] > 1:
                        imbalance_number += 1
                total_imbalance += imbalance_number
        return total_imbalance