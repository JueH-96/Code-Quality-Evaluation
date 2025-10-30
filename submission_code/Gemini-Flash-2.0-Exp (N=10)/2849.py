class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0

        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                sorted_sub_array = sorted(sub_array)
                imbalance = 0
                for k in range(len(sorted_sub_array) - 1):
                    if sorted_sub_array[k+1] - sorted_sub_array[k] > 1:
                        imbalance += 1
                total_imbalance += imbalance
        
        return total_imbalance