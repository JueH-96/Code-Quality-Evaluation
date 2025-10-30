class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                sorted_subarray = sorted(subarray)
                imbalance = 0
                if len(sorted_subarray) > 1:
                    for k in range(len(sorted_subarray) - 1):
                        if sorted_subarray[k+1] - sorted_subarray[k] > 1:
                            imbalance += 1
                total_imbalance += imbalance
        return total_imbalance