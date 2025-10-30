class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = 0
        
        for i in range(n - 1):
            # check start of potential alternating subarray
            if nums[i + 1] != nums[i] + 1:
                continue
            
            curr_len = 2
            expected_diff = -1  # next difference should be -1
            j = i + 2
            
            while j < n and nums[j] - nums[j - 1] == expected_diff:
                curr_len += 1
                expected_diff *= -1  # flip between +1 and -1
                j += 1
            
            max_len = max(max_len, curr_len)
        
        return max_len if max_len > 1 else -1