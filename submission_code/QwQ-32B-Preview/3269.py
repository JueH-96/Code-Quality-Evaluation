class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        count = 0
        
        # Iterate through all possible starting indices for subarrays of length m+1
        for i in range(n - m):
            match = True
            # Check each pattern element
            for k in range(m):
                if pattern[k] == 1:
                    if not (nums[i + k + 1] > nums[i + k]):
                        match = False
                        break
                elif pattern[k] == 0:
                    if not (nums[i + k + 1] == nums[i + k]):
                        match = False
                        break
                elif pattern[k] == -1:
                    if not (nums[i + k + 1] < nums[i + k]):
                        match = False
                        break
            if match:
                count += 1
        return count