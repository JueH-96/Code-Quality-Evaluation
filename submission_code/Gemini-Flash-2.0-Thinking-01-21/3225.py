from collections import Counter

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        max_len = 0
        left = 0
        freq_map = Counter()
        for right in range(len(nums)):
            freq_map[nums[right]] += 1
            while freq_map[nums[right]] > k:
                freq_map[nums[left]] -= 1
                if freq_map[nums[left]] == 0:
                    del freq_map[nums[left]]
                left += 1
            max_len = max(max_len, right - left + 1)
        return max_len