from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        freq = {}
        left = 0
        max_length = 0
        count_over_k = 0  # Number of elements with frequency > k

        for right in range(len(nums)):
            # Include nums[right] in the window
            freq[nums[right]] = freq.get(nums[right], 0) + 1
            if freq[nums[right]] == k + 1:
                count_over_k += 1

            # Shrink the window from the left if any frequency exceeds k
            while count_over_k > 0:
                freq[nums[left]] -= 1
                if freq[nums[left]] == k:
                    count_over_k -= 1
                left += 1

            # Update the maximum length found
            current_window_length = right - left + 1
            if current_window_length > max_length:
                max_length = current_window_length

        return max_length