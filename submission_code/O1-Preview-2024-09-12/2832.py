class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        counts = defaultdict(int)
        max_count = 0
        left = 0
        result = 0
        for right in range(len(nums)):
            counts[nums[right]] += 1
            max_count = max(max_count, counts[nums[right]])
            while (right - left + 1) - max_count > k:
                counts[nums[left]] -= 1
                left += 1
            result = max(result, max_count)
        return result