from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        total_count = Counter(nums)
        dominant = max(total_count, key=total_count.get)
        left_count = 0
        for i in range(len(nums) - 1):
            if total_count[dominant] * 2 <= len(nums):
                break
            if nums[i] == dominant:
                left_count += 1
            total_count[nums[i]] -= 1
            if left_count * 2 > i + 1 and (total_count[dominant] + left_count) * 2 > len(nums):
                return i
        return -1