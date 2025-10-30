class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0
        
        for i in range(n):
            sarr = sorted(nums[:i+1])
            imbalance = 0
            for j in range(1, len(sarr)):
                if sarr[j] - sarr[j-1] > 1:
                    imbalance += 1
            total_imbalance += imbalance
        
        return total_imbalance