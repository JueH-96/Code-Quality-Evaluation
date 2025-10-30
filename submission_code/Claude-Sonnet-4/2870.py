class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        n = len(nums)
        
        for i in range(n - 1):
            # Start a new alternating subarray from position i
            length = 1
            expected_diff = 1  # First difference should be +1
            
            for j in range(i, n - 1):
                actual_diff = nums[j + 1] - nums[j]
                
                if actual_diff == expected_diff:
                    length += 1
                    # Alternate the expected difference for next iteration
                    expected_diff = -expected_diff
                else:
                    break
            
            # Only consider subarrays of length > 1
            if length > 1:
                max_length = max(max_length, length)
        
        return max_length