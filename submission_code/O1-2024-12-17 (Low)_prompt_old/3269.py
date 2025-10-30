class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n, m = len(nums), len(pattern)
        count = 0
        
        # We need subarrays of length m+1
        # The last valid starting index is n-(m+1)
        for i in range(n - m):
            # Check if subarray nums[i..i+m] matches the pattern
            match = True
            for k in range(m):
                # Compare nums[i + k] and nums[i + k + 1]
                if pattern[k] == 1:
                    if not (nums[i + k + 1] > nums[i + k]):
                        match = False
                        break
                elif pattern[k] == 0:
                    if not (nums[i + k + 1] == nums[i + k]):
                        match = False
                        break
                else:  # pattern[k] == -1
                    if not (nums[i + k + 1] < nums[i + k]):
                        match = False
                        break
            
            if match:
                count += 1
        
        return count