class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        count = 0
        
        # Check each possible starting position for a subarray of size m+1
        for i in range(n - m):
            matches = True
            
            # Check if the subarray starting at i matches the pattern
            for k in range(m):
                if pattern[k] == 1:
                    if nums[i + k + 1] <= nums[i + k]:
                        matches = False
                        break
                elif pattern[k] == 0:
                    if nums[i + k + 1] != nums[i + k]:
                        matches = False
                        break
                elif pattern[k] == -1:
                    if nums[i + k + 1] >= nums[i + k]:
                        matches = False
                        break
            
            if matches:
                count += 1
        
        return count