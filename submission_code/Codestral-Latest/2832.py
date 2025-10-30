class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict

        n = len(nums)
        freq = defaultdict(int)
        max_len = 0
        left = 0

        for right in range(n):
            freq[nums[right]] += 1
            max_freq = max(freq.values())

            while (right - left + 1) - max_freq > k:
                freq[nums[left]] -= 1
                left += 1

            max_len = max(max_len, max_freq)

        return max_len