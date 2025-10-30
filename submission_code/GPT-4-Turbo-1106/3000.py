class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        nums_sorted = sorted((num, i) for i, num in enumerate(nums))
        min_diff = float('inf')
        left = 0
        
        for right in range(len(nums_sorted)):
            while nums_sorted[right][1] - nums_sorted[left][1] >= x:
                min_diff = min(min_diff, abs(nums_sorted[right][0] - nums_sorted[left][0]))
                left += 1
        
        return min_diff