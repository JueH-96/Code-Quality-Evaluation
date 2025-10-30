class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return -1
        
        ans = -1
        curr_len = 1
        expected_diff = 1  # The first required difference (+1).
        
        for i in range(1, n):
            diff = nums[i] - nums[i - 1]
            if diff == expected_diff:
                # Continue the alternating pattern
                curr_len += 1
                # Flip the expected difference (+1 -> -1 or -1 -> +1)
                expected_diff *= -1
            else:
                # Pattern broken, try to start a new pattern if diff == +1
                if diff == 1:
                    curr_len = 2
                    expected_diff = -1
                else:
                    curr_len = 1
                    expected_diff = 1
            
            # Update answer only if current length > 1
            if curr_len > 1:
                ans = max(ans, curr_len)
        
        return ans