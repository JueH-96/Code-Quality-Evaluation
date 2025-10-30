class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        count = 0
        n = len(nums)
        m = len(pattern)
        
        for i in range(n - m):
            matches = True
            for k in range(m):
                if pattern[k] == 1 and nums[i + k + 1] <= nums[i + k]:
                    matches = False
                    break
                if pattern[k] == 0 and nums[i + k + 1] != nums[i + k]:
                    matches = False
                    break
                if pattern[k] == -1 and nums[i + k + 1] >= nums[i + k]:
                    matches = False
                    break
            if matches:
                count += 1
        
        return count