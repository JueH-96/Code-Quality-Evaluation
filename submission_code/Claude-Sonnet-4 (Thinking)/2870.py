class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = -1
        
        for i in range(n - 1):
            if nums[i + 1] - nums[i] == 1:  # Can start an alternating subarray
                length = 2
                expected_diff = -1  # Next difference should be -1
                
                for j in range(i + 2, n):
                    if nums[j] - nums[j - 1] == expected_diff:
                        length += 1
                        expected_diff = -expected_diff  # Alternate between 1 and -1
                    else:
                        break
                
                max_len = max(max_len, length)
        
        return max_len