class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        m = len(pattern)
        n = len(nums)
        count = 0
        
        # Iterate over each possible starting index for a subarray of length m+1
        for i in range(n - m):
            matches = True
            # Check each condition in the pattern
            for k in range(m):
                if (pattern[k] == 1 and not (nums[i + k + 1] > nums[i + k])) or \
                   (pattern[k] == 0 and not (nums[i + k + 1] == nums[i + k])) or \
                   (pattern[k] == -1 and not (nums[i + k + 1] < nums[i + k])):
                    matches = False
                    break
            if matches:
                count += 1
        
        return count