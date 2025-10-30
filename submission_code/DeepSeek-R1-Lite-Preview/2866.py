class Solution:
    def longestAlternatingSubarray(self, nums, threshold):
        max_length = 0
        n = len(nums)
        
        for i in range(n):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                current_length = 1
                expected_parity = 1  # Next should be odd
                for j in range(i + 1, n):
                    if nums[j] <= threshold and nums[j] % 2 == expected_parity:
                        current_length += 1
                        expected_parity = 1 - expected_parity  # Toggle parity
                    else:
                        break
                if current_length > max_length:
                    max_length = current_length
        return max_length