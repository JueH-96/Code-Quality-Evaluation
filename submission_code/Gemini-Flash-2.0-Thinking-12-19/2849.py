class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                sarr = sorted(subarray)
                imbalance_count = 0
                for k in range(len(sarr) - 1):
                    if sarr[k+1] - sarr[k] > 1:
                        imbalance_count += 1
                total_imbalance += imbalance_count
        return total_imbalance