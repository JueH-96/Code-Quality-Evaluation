class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        found = False
        
        # Precompute the minimum value to the left of each index
        left_min = [0] * n
        left_min[0] = nums[0]
        for i in range(1, n):
            left_min[i] = min(left_min[i-1], nums[i])
        
        # Precompute the minimum value to the right of each index
        right_min = [0] * n
        right_min[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            right_min[i] = min(right_min[i+1], nums[i])
        
        # Find the minimum sum of a mountain triplet
        for j in range(1, n-1):
            if nums[j] > left_min[j-1] and nums[j] > right_min[j+1]:
                found = True
                min_sum = min(min_sum, left_min[j-1] + nums[j] + right_min[j+1])
        
        return min_sum if found else -1