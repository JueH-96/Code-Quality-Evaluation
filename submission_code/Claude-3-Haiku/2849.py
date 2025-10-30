class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0
        
        for i in range(n):
            for j in range(i, n):
                subarray = sorted(nums[i:j+1])
                imbalance = 0
                for k in range(len(subarray)-1):
                    if subarray[k+1] - subarray[k] > 1:
                        imbalance += 1
                total_imbalance += imbalance
        
        return total_imbalance