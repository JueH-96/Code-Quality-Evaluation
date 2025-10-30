class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        
        for j in range(1, n - 1):
            left_min = float('inf')
            for i in range(j):
                if nums[i] < nums[j]:
                    left_min = min(left_min, nums[i])
            
            right_min = float('inf')
            for k in range(j + 1, n):
                if nums[k] < nums[j]:
                    right_min = min(right_min, nums[k])
            
            if left_min != float('inf') and right_min != float('inf'):
                min_sum = min(min_sum, left_min + nums[j] + right_min)
        
        if min_sum == float('inf'):
            return -1
        else:
            return min_sum