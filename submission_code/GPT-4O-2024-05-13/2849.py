class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        def imbalance_number(arr):
            sarr = sorted(arr)
            imbalance = 0
            for i in range(len(sarr) - 1):
                if sarr[i + 1] - sarr[i] > 1:
                    imbalance += 1
            return imbalance
        
        n = len(nums)
        total_imbalance = 0
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j + 1]
                total_imbalance += imbalance_number(subarray)
        
        return total_imbalance