class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        sum_imbalance = 0
        
        # Function to calculate the imbalance number of a sorted subarray
        def imbalance(sarr):
            count = 0
            for i in range(len(sarr) - 1):
                if sarr[i + 1] - sarr[i] > 1:
                    count += 1
            return count
        
        # Iterate over all possible subarrays
        for i in range(n):
            for j in range(i + 1, n + 1):
                subarray = sorted(nums[i:j])
                sum_imbalance += imbalance(subarray)
        
        return sum_imbalance