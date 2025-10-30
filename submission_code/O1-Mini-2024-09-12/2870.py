class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        
        for i in range(1, n):
            if nums[i] == nums[i - 1] + 1:
                current_len = 2
                prev_diff = 1
                j = i + 1
                while j < n:
                    current_diff = nums[j] - nums[j - 1]
                    expected_diff = -prev_diff
                    if current_diff == expected_diff:
                        current_len += 1
                        prev_diff = current_diff
                        j += 1
                    else:
                        break
                if current_len > max_len:
                    max_len = current_len
                    
        return max_len