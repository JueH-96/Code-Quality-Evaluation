from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        max_len = 0
        freq = defaultdict(int)
        left = 0
        for right in range(len(nums)):
            x = nums[right]
            freq[x] += 1
            # Calculate the number of non-x elements in the current window
            non_x = sum(freq.values()) - freq[x]
            # Ensure non_x <= k by moving left pointer
            while non_x > k:
                y = nums[left]
                freq[y] -= 1
                left += 1
            # Update the maximum length
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        return max_len