class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        m = n // 2
        cost = 0
        
        if nums[m] < k:
            # Need to increase elements from m to end that are less than k
            for i in range(m, n):
                if nums[i] < k:
                    cost += k - nums[i]
        else:
            # Need to decrease elements from 0 to m that are greater than k
            for i in range(m + 1):
                if nums[i] > k:
                    cost += nums[i] - k
        
        return cost