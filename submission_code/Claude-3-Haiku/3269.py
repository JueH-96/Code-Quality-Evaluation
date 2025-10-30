class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n, m = len(nums), len(pattern)
        count = 0
        
        for i in range(n - m + 1):
            if self.isMatch(nums, i, pattern):
                count += 1
        
        return count
    
    def isMatch(self, nums: List[int], start: int, pattern: List[int]) -> bool:
        for i, p in enumerate(pattern):
            if p == 1 and nums[start + i + 1] <= nums[start + i]:
                return False
            elif p == 0 and nums[start + i + 1] != nums[start + i]:
                return False
            elif p == -1 and nums[start + i + 1] >= nums[start + i]:
                return False
        
        return True