from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        left = 0
        max_len = 0
        count = {}

        for right in range(len(nums)):
            if nums[right] not in count:
                count[nums[right]] = 0
            count[nums[right]] += 1

            while (right - left + 1) - max(count.values()) > k:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    del count[nums[left]]
                left += 1

            max_len = max(max_len, right - left + 1)

        return max_len - (max_len - max(count.values()))