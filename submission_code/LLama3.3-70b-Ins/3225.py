from typing import List

class Solution:
    def longestSubarray(self, nums: List[int], k: int) -> int:
        def atMost(k: int) -> int:
            if k < 1:
                return 0
            left = 0
            count = {}
            max_length = 0
            for right in range(len(nums)):
                if nums[right] not in count:
                    count[nums[right]] = 0
                count[nums[right]] += 1
                while max(count.values()) > k:
                    count[nums[left]] -= 1
                    if count[nums[left]] == 0:
                        del count[nums[left]]
                    left += 1
                max_length = max(max_length, right - left + 1)
            return max_length

        return atMost(k)