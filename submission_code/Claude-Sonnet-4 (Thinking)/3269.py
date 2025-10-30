class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        count = 0
        
        # Check each possible starting position for a subarray of size m+1
        for i in range(n - m):
            match = True
            # Check each pattern element
            for k in range(m):
                left, right = nums[i + k], nums[i + k + 1]
                
                # Convert the relationship to pattern format
                if right > left:
                    actual = 1
                elif right == left:
                    actual = 0
                else:  # right < left
                    actual = -1
                
                # If actual relationship doesn't match expected pattern
                if actual != pattern[k]:
                    match = False
                    break
            
            if match:
                count += 1
        
        return count