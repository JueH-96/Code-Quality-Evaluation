class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        result, minSum = float('inf'), float('inf')
        left_min, right_min = [0]*len(nums), [0]*len(nums)
        
        left_min[0] = nums[0]
        for i in range(1, len(nums)):
            left_min[i] = min(left_min[i-1], nums[i])
        
        right_min[-1] = nums[-1]
        for i in range(len(nums)-2, -1, -1):
            right_min[i] = min(right_min[i+1], nums[i])
        
        left, right = 0, 2
        while right < len(nums):
            if left_min[left] < nums[right] and right_min[right] < nums[right]:
                minSum = nums[left] + nums[right] + nums[right-1]
                result = min(result, minSum)
            left += 1
            right += 1
        return result if result != float('inf') else -1