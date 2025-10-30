class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        counter = defaultdict(int)
        left = 0
        max_length = 0
        for right in range(len(nums)):
            counter[nums[right]] += 1
            while (right - left + 1) - max(counter.values()) > k:
                counter[nums[left]] -= 1
                left += 1
            max_length = max(max_length, right - left + 1)
        return max_length