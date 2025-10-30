class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        left_min = [0] * n
        right_min = [0] * n
        left_min[0] = nums[0]
        right_min[n-1] = nums[n-1]
        
        # Fill left_min array
        for i in range(1, n):
            left_min[i] = min(left_min[i-1], nums[i])
        
        # Fill right_min array
        for i in range(n-2, -1, -1):
            right_min[i] = min(right_min[i+1], nums[i])
        
        # Find minimum mountain triplet sum
        min_sum = float('inf')
        for i in range(1, n-1):
            if left_min[i-1] < nums[i] > right_min[i+1]:
                min_sum = min(min_sum, nums[i] + left_min[i-1] + right_min[i+1])
        
        return min_sum if min_sum != float('inf') else -1