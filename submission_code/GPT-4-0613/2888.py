class Solution:
    def minimumIndex(self, nums):
        from collections import Counter
        count = Counter(nums)
        dominant = max(count, key=count.get)
        left_count = Counter()
        for i, num in enumerate(nums):
            left_count[num] += 1
            if left_count[dominant] * 2 > i + 1 and (count[dominant] - left_count[dominant]) * 2 > len(nums) - i - 1:
                return i + 1
        return -1