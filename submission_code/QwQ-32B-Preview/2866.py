class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        for i in range(len(nums)):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                current_length = 1
                expected_parity = 1  # Next should be odd
                j = i + 1
                while j < len(nums):
                    if nums[j] <= threshold and nums[j] % 2 == expected_parity:
                        current_length += 1
                        expected_parity = 1 - expected_parity  # Alternate parity
                        j += 1
                    else:
                        break
                if current_length > max_length:
                    max_length = current_length
        return max_length