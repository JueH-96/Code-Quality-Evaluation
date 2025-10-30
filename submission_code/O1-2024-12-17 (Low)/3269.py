class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        count = 0
        
        # We want subarrays of length m+1, so we can iterate i up to n-(m+1)
        for i in range(n - (m + 1) + 1):
            match = True
            # Check if the subarray nums[i : i+m+1] matches the pattern
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