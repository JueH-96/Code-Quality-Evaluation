class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = n // 2
        
        if nums[median_index] == k:
            return 0
        
        operations = 0
        if nums[median_index] < k:
            for i in range(median_index, n):
                if nums[i] >= k:
                    break
                operations += k - nums[i]
        else:
            for i in range(median_index, -1, -1):
                if nums[i] <= k:
                    break
                operations += nums[i] - k
        
        return operations