class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        m = len(pattern)
        count = 0
        
        for i in range(len(nums) - m):
            match = True
            for k in range(m):
                if pattern[k] == 1 and not (nums[i + k + 1] > nums[i + k]):
                    match = False
                    break
                elif pattern[k] == 0 and not (nums[i + k + 1] == nums[i + k]):
                    match = False
                    break
                elif pattern[k] == -1 and not (nums[i + k + 1] < nums[i + k]):
                    match = False
                    break
            
            if match:
                count += 1
        
        return count