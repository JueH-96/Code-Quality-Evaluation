class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        count = 0
        
        # Iterate over each possible starting index for the subarray
        for i in range(n - m):
            match = True
            # Check if the subarray starting at index i matches the pattern
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