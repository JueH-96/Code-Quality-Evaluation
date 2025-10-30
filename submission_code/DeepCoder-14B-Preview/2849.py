class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        total = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                sub = nums[i:j+1]
                sarr = sorted(sub)
                imbalance = sum(1 for k in range(len(sarr)-1) if sarr[k+1] - sarr[k] > 1)
                total += imbalance
        return total