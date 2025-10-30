class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Precompute left minimums
        left_min = [float('inf')] * n
        for j in range(1, n):
            left_min[j] = left_min[j-1]
            if nums[j-1] < nums[j]:
                left_min[j] = min(left_min[j], nums[j-1])
        
        # Precompute right minimums
        right_min = [float('inf')] * n
        for j in range(n-2, -1, -1):
            right_min[j] = right_min[j+1]
            if nums[j+1] < nums[j]:
                right_min[j] = min(right_min[j], nums[j+1])
        
        # Find minimum sum
        min_sum = float('inf')
        for j in range(1, n-1):
            if left_min[j] != float('inf') and right_min[j] != float('inf'):
                min_sum = min(min_sum, left_min[j] + nums[j] + right_min[j])
        
        return min_sum if min_sum != float('inf') else -1