class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        min_diff = float('inf')
        
        for i in range(1, n):
            diff = abs(max(nums[:i]) - min(nums[i:]))
            min_diff = min(min_diff, diff)
        
        return min_diff