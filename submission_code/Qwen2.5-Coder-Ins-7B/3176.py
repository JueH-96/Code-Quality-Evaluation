class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return -1
        
        left_min = [float('inf')] * n
        right_min = [float('inf')] * n
        
        left_min[0] = nums[0]
        for i in range(1, n):
            left_min[i] = min(left_min[i-1], nums[i])
        
        right_min[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            right_min[i] = min(right_min[i+1], nums[i])
        
        min_sum = float('inf')
        for j in range(1, n-1):
            if nums[j] > left_min[j] and nums[j] > right_min[j]:
                min_sum = min(min_sum, left_min[j] + nums[j] + right_min[j])
        
        return min_sum if min_sum != float('inf') else -1