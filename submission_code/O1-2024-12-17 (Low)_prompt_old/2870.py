class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = -1
        
        for start in range(n - 1):
            # We need s[1] == s[0] + 1 to begin a valid alternating subarray
            if nums[start+1] == nums[start] + 1:
                # We have at least a subarray of length 2
                length = 2
                # Next expected difference is -1
                next_diff = -1
                
                # Extend the subarray as far as it follows the pattern
                for i in range(start+1, n-1):
                    # Check if the difference is what we expect
                    if nums[i+1] - nums[i] == next_diff:
                        length += 1
                        # Flip the expected difference
                        next_diff *= -1
                    else:
                        break
                
                # Update the maximum length
                max_len = max(max_len, length)
        
        return max_len