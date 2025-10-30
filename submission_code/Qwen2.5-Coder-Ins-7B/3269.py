class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        count = 0
        m = len(pattern)
        n = len(nums)
        
        for i in range(n - m):
            match = True
            for j in range(m):
                if pattern[j] == 1 and nums[i + j + 1] <= nums[i + j]:
                    match = False
                    break
                elif pattern[j] == 0 and nums[i + j + 1] != nums[i + j]:
                    match = False
                    break
                elif pattern[j] == -1 and nums[i + j + 1] >= nums[i + j]:
                    match = False
                    break
            if match:
                count += 1
        
        return count