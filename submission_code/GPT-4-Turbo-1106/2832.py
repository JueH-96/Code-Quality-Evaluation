class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict

        # Helper function to find the longest subarray with at most k different elements
        def longest_subarray_with_k_deletions(nums, k):
            max_length = 0
            left = 0
            freq = defaultdict(int)
            for right in range(len(nums)):
                freq[nums[right]] += 1
                # If we have more than k elements to delete, move the left pointer
                while (right - left + 1) - max(freq.values()) > k:
                    freq[nums[left]] -= 1
                    left += 1
                max_length = max(max_length, right - left + 1)
            return max_length

        return longest_subarray_with_k_deletions(nums, k)