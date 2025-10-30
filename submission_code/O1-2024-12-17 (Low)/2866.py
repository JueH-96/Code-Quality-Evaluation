class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        i = 0
        n = len(nums)

        while i < n:
            # We only try to start a subarray if nums[i] is even and <= threshold
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                length = 1
                j = i + 1
                # Expand as long as parity alternates and value <= threshold
                while j < n and (nums[j] % 2 != nums[j - 1] % 2) and nums[j] <= threshold:
                    length += 1
                    j += 1
                max_len = max(max_len, length)
            i += 1

        return max_len