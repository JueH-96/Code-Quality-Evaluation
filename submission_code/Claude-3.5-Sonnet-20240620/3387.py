class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        median_index = n // 2
        nums.sort()
        
        if n % 2 == 0:
            median_index -= 1
        
        operations = 0
        left, right = median_index, median_index
        
        while left >= 0 and right < n:
            left_diff = abs(nums[left] - k)
            right_diff = abs(nums[right] - k)
            
            operations += left_diff + right_diff
            
            if left == right:
                return operations
            
            left -= 1
            right += 1
        
        while left >= 0:
            operations += abs(nums[left] - k)
            left -= 1
        
        while right < n:
            operations += abs(nums[right] - k)
            right += 1
        
        return operations