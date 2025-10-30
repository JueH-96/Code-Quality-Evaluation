class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        def imbalance_number(subarr):
            sarr = sorted(subarr)
            imbalance = 0
            for i in range(len(sarr) - 1):
                if sarr[i + 1] - sarr[i] > 1:
                    imbalance += 1
            return imbalance
        
        n = len(nums)
        total_imbalance = 0
        
        for start in range(n):
            for end in range(start + 1, n + 1):
                subarr = nums[start:end]
                total_imbalance += imbalance_number(subarr)
        
        return total_imbalance