class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        mid = n // 2
        cost = 0
        
        for i in range(mid):
            cost += max(0, nums[i] - k)
        
        cost += abs(nums[mid] - k)
        
        for i in range(mid + 1, n):
            cost += max(0, k - nums[i])
        
        return cost