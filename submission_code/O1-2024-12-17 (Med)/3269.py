class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        count = 0
        
        # Iterate over all possible starting indices for subarrays of length m+1
        for start in range(n - m):
            match_found = True
            # Check each relation defined by the pattern
            for k in range(m):
                if pattern[k] == 1:
                    if not (nums[start + k + 1] > nums[start + k]):
                        match_found = False
                        break
                elif pattern[k] == 0:
                    if not (nums[start + k + 1] == nums[start + k]):
                        match_found = False
                        break
                elif pattern[k] == -1:
                    if not (nums[start + k + 1] < nums[start + k]):
                        match_found = False
                        break
            if match_found:
                count += 1
        
        return count