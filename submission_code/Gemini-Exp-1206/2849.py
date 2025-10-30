class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                sorted_sub_array = sorted(sub_array)
                imbalance_number = 0
                for k in range(len(sorted_sub_array) - 1):
                    if sorted_sub_array[k+1] - sorted_sub_array[k] > 1:
                        imbalance_number += 1
                ans += imbalance_number
        return ans