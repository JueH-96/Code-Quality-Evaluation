class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        result = float('inf')
        
        # Create a sorted list of (value, index) pairs
        sorted_nums = sorted((num, i) for i, num in enumerate(nums))
        
        left = 0
        for right in range(x, n):
            while left < n and sorted_nums[left][1] <= sorted_nums[right][1] - x:
                result = min(result, abs(sorted_nums[right][0] - sorted_nums[left][0]))
                left += 1
        
        return result