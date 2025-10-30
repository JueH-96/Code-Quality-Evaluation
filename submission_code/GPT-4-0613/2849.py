class Solution:
    def sumImbalanceNumbers(self, nums):
        n = len(nums)
        total = 0
        for i in range(n):
            for j in range(i+1, n+1):
                subarray = sorted(nums[i:j])
                imbalance = 0
                for k in range(len(subarray)-1):
                    if subarray[k+1] - subarray[k] > 1:
                        imbalance += 1
                total += imbalance
        return total